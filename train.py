import os
import json
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from operator import itemgetter
import torch
from torch.amp import autocast, GradScaler
import wandb

from utils import set_random_seed, get_current_time
from utils.metrics import metric_func
from data.dataset import RegionLevelSTDataset, region_collate_fn
from data.normalize_utils import get_normalize_method
from models.rma import RetentionMoEPredictor, ModelConfig

@torch.no_grad()
def evaluate(model, dataloader, device, genes, return_all=False, use_amp=True):
    model.eval()
    all_preds = []
    all_labels = []
    
    for batch in dataloader:
        features, coords, gene_exp, masks = [x.to(device) for x in batch]
        if use_amp:
            with autocast(device_type="cuda"):
                pred = model.inference(features, coords, masks)
        else:
            pred = model.inference(features, coords, masks)
        
        for b in range(pred.shape[0]):
            n_valid = masks[b].sum().item()
            all_preds.append(pred[b, :n_valid].detach().cpu().numpy())
            all_labels.append(gene_exp[b, :n_valid].detach().cpu().numpy())
        

        del features, coords, gene_exp, masks, pred
    
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    fold_metrics = metric_func(all_preds, all_labels, genes)

    if return_all:
        return fold_metrics, {'preds_all': all_preds, 'targets_all': all_labels}
    else:
        return fold_metrics


def main(args, split_id, train_sample_ids, test_sample_ids, val_save_dir, checkpoint_save_dir):
    normalize_method = args.normalize_method
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    print(f"--- Processing Split {split_id} ---")
    print("Dataset Loading...")
    embed_root = os.path.join(args.embed_dataroot, args.dataset, args.feature_encoder)
    st_root = os.path.join(args.source_dataroot, args.dataset, "adata")
    gene_list_path = os.path.join(args.gene_list_dir, args.dataset, args.gene_list)

    train_dataset = RegionLevelSTDataset(
        sample_ids=train_sample_ids,
        embed_root=embed_root,
        st_root=st_root,
        gene_list_path=gene_list_path,
        normalize_method=normalize_method,
        distribution="uniform",
        sample_times=args.sample_times
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, 
        num_workers=args.num_workers, collate_fn=region_collate_fn(),
        pin_memory=True,
        persistent_workers=False )
    
    test_dataset = RegionLevelSTDataset(
        sample_ids=test_sample_ids,
        embed_root=embed_root,
        st_root=st_root,
        gene_list_path=gene_list_path,
        normalize_method=normalize_method,
        distribution="constant_1.0",
        sample_times=1
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, 
        num_workers=args.num_workers, collate_fn=region_collate_fn()
    )
    
    actual_n_genes = len(train_dataset.genes)
    print(f"Actual n_genes: {actual_n_genes}, n_macrogenes: {args.n_macrogenes}")
    args.n_genes = actual_n_genes

    config = ModelConfig(
        n_genes=actual_n_genes,
        n_macrogenes=args.n_macrogenes,
        feature_dim=args.feature_dim,
        d_model=args.hidden_dim,
        dropout=args.dropout,
        n_heads=args.n_heads,
        n_retention_layers=args.n_retention_layers,
        retention_init_value=args.retention_init_value,
        retention_heads_range=args.retention_heads_range,
        ffn_ratio=args.ffn_ratio,
        drop_path_rate=args.drop_path_rate,
        top_k=args.top_k,
        d_latent=args.d_latent,
        n_router_heads=args.n_router_heads,
        mse_weight=args.mse_weight,
        pcc_weight=args.pcc_weight,
        attention_chunk_size=getattr(args, 'attention_chunk_size', None),
        expert_type=getattr(args, 'expert_type', 'mlp_latent'),
        global_weight=getattr(args, 'global_weight', 0.3),
        group_mask_eps=getattr(args, 'group_mask_eps', 0.0),
    )

    model = RetentionMoEPredictor(config).to(device)

    if args.protein_embedding_path:
        gene_names = train_dataset.genes
        print(f"Loading protein embeddings for {len(gene_names)} genes...")
        try:
            model.load_protein_embeddings(
                args.protein_embedding_path, 
                args.gene_emb_dim,
                gene_names, 
                device,
                cluster_method=args.cluster_method,
                use_gene_group_mask=getattr(args, "use_gene_group_mask", False),
            )
        except Exception as e:
            print(f"Warning: Failed to load protein embeddings: {e}")
        
        stats = model.get_group_statistics()
        if stats:
            print(f"Gene grouping: {stats['n_groups']} groups, sizes: {stats['min_size']}-{stats['max_size']}")

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model initialized. Trainable Params: {trainable_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    use_amp = bool(getattr(args, "use_amp", True)) and device.type == "cuda"
    scaler = GradScaler() if use_amp else None

    print("Start Training...")
    if use_amp:
        print("Mixed precision training (AMP) enabled")
    best_pearson = -1.0
    early_stop_step = 0
    best_fold_metrics = None
    eps_progress = None
    epoch_iter = tqdm(range(1, args.epochs + 1), ncols=100)

    for epoch in epoch_iter:
        avg_loss = 0
        model.train()

        try:

            fusion_w = model.moe_head.get_fusion_weights()
            moe_weight = float(fusion_w[0].cpu().item())  
            print(f"[Split {split_id}] epoch={epoch} moe_weight={moe_weight:.6f}")
            if args.use_wandb:
                wandb.log({f"{args.dataset}/{split_id}/Train/moe_weight": moe_weight, "epoch": epoch})
        except Exception:
            pass

        # Group-mask eps schedule (only when enabled)
        if getattr(args, "use_gene_group_mask", False):
            eps_start = float(getattr(args, "group_mask_eps_start", 0.8))
            eps_end = float(getattr(args, "group_mask_eps_end", 0.1))

            # 1) Base linear annealing by epoch
            decay_epochs = int(getattr(args, "group_mask_eps_decay_epochs", 0)) or int(args.epochs)
            decay_epochs = max(decay_epochs, 2)
            base_t = (epoch - 1) / (decay_epochs - 1)
            base_t = float(min(max(base_t, 0.0), 1.0))

            patience = int(getattr(args, "earlystopping", 1) or 1)
            plateau_bonus = float(min(max(early_stop_step / float(patience), 0.0), 1.0))  # in [0,1]


            t_candidate = float(min(max(base_t + plateau_bonus, 0.0), 1.0))


            eps_candidate = eps_start + (eps_end - eps_start) * t_candidate


            if eps_progress is None:
                eps_progress = float(eps_start)


            eps_progress = float(min(eps_progress, eps_candidate))


            lo = float(min(eps_start, eps_end))
            hi = float(max(eps_start, eps_end))
            eps = float(min(max(eps_progress, lo), hi))
            try:
                model.moe_head.set_group_mask_eps(eps)
                print(
                    f"[Split {split_id}] epoch={epoch} group_mask_eps={eps:.6f} "
                    f"(base_t={base_t:.3f}, bonus={plateau_bonus:.3f}, cand={eps_candidate:.6f})"
                )
                if args.use_wandb:
                    wandb.log({
                        f"{args.dataset}/{split_id}/Train/group_mask_eps": eps,
                        f"{args.dataset}/{split_id}/Train/group_mask_eps_candidate": float(eps_candidate),
                        f"{args.dataset}/{split_id}/Train/group_mask_base_t": float(base_t),
                        f"{args.dataset}/{split_id}/Train/group_mask_plateau_bonus": float(plateau_bonus),
                        "epoch": epoch
                    })
            except Exception:
                pass


        for step, batch in enumerate(train_loader):
            batch = [x.to(device) for x in batch]
            features, coords, genes, masks = batch
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast(device_type="cuda"):
                    pred, loss = model(features, coords, genes, masks)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                pred, loss = model(features, coords, genes, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
                optimizer.step()

            loss_value = loss.detach().cpu().item()
            
            if args.use_wandb:
                wandb.log({f"{args.dataset}/{split_id}/Train/loss": loss_value, "epoch": epoch})
            avg_loss += loss_value
            
            del batch, features, coords, genes, masks, pred, loss

        avg_loss /= len(train_loader)
        epoch_iter.set_description(f"epoch: {epoch}, loss: {avg_loss:.4f}")


        if args.save_step > 0 and epoch % args.save_step == 0:
            torch.save(model.state_dict(), os.path.join(checkpoint_save_dir, f"{epoch}.pth"))

        if epoch % args.eval_step == 0 or epoch == args.epochs:
            fold_metrics, pred_dict = evaluate(model, test_loader, device, train_dataset.genes, return_all=True, use_amp=use_amp)
            current_pearson = fold_metrics["metrics"]['pearson_mean']

            if current_pearson > best_pearson:
                best_pearson = current_pearson
                best_fold_metrics = fold_metrics
                
                with open(os.path.join(val_save_dir, f'{split_id}_results.json'), 'w') as f:
                    json.dump(best_fold_metrics, f, indent=4)
                
                if args.save_macrogene_info:
                    save_macrogene_info(model, args, val_save_dir, split_id)

                save_path = os.path.join(val_save_dir, 'predictions.npz')
                np.savez_compressed(
                    save_path,
                    preds_all=pred_dict['preds_all'],
                    targets_all=pred_dict['targets_all']
                )
                
                early_stop_step = 0
                print(f"Epoch {epoch} - Test Pearson: {current_pearson:.4f}")
            else:
                early_stop_step += 1
                if early_stop_step >= args.earlystopping:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if args.use_wandb:
                wandb.log({
                    f"{args.dataset}/{split_id}/Val/pearson_mean": fold_metrics["metrics"]['pearson_mean'],
                    f"{args.dataset}/{split_id}/Val/mse_mean": fold_metrics["metrics"]['mse_mean'],
                    f"{args.dataset}/{split_id}/Val/best_pearson": best_pearson,
                    "epoch": epoch
                })
            
            # Fix: clean up evaluation results to free memory
            del fold_metrics, pred_dict
            torch.cuda.empty_cache()
    
    torch.cuda.empty_cache()
    return best_fold_metrics


def save_macrogene_info(model, args, save_dir, split_id):
    try:
        gene_to_group = model.gene_grouper.gene_to_group.detach().cpu()
        torch.save({
            'gene_to_group': gene_to_group,
            'n_macrogenes': args.n_macrogenes,
        }, os.path.join(save_dir, f'{split_id}_gene_groups.pt'))
        
        weights = model.get_macrogene_weights()
        if weights is not None:
            torch.save(weights.detach().cpu(), 
                       os.path.join(save_dir, f'{split_id}_macrogene_weights.pt'))
    except Exception as e:
        print(f"Warning: Failed to save macrogene info: {e}")


def run(args):
    """Run cross-validation across multiple splits."""
    split_dir = os.path.join(args.source_dataroot, args.dataset, 'splits')
    if not os.path.exists(split_dir):
        print(f"Error: Split directory not found: {split_dir}")
        return None

    splits = sorted([x for x in os.listdir(split_dir) if 'train_' in x])
    
    all_metrics = []
    
    for i in range(len(splits)):
        train_df = pd.read_csv(os.path.join(split_dir, f'train_{i}.csv'))
        test_df = pd.read_csv(os.path.join(split_dir, f'test_{i}.csv'))

        train_sample_ids = train_df['sample_id'].tolist()
        test_sample_ids = test_df['sample_id'].tolist()

        kfold_save_dir = os.path.join(args.save_dir, f'split{i}')
        os.makedirs(kfold_save_dir, exist_ok=True)
        
        checkpoint_save_dir = os.path.join(kfold_save_dir, 'checkpoints')
        if args.save_step > 0:
            os.makedirs(checkpoint_save_dir, exist_ok=True)

        fold_result = main(args, i, train_sample_ids, test_sample_ids, kfold_save_dir, checkpoint_save_dir)

        if fold_result:
            all_metrics.append({'metrics': fold_result['metrics'], 'quantiles': fold_result['quantiles']})

    if all_metrics:
        summary = {
            'dataset': args.dataset,
            'method': args.model,
            'n_splits': len(all_metrics),
            'n_macrogenes': args.n_macrogenes,
            'pearson_mean': float(np.mean([m['metrics']['pearson_mean'] for m in all_metrics])),
            'pearson_std': float(np.mean([m['metrics']['pearson_std'] for m in all_metrics])),
            'pearson_median': float(np.mean([m['quantiles']['pearson_q2'] for m in all_metrics])),
            'r2_mean': float(np.mean([m['metrics']['r2_mean'] for m in all_metrics])),
            'r2_std': float(np.mean([m['metrics']['r2_std'] for m in all_metrics])),
            'r2_median': float(np.mean([m['quantiles']['r2_q2'] for m in all_metrics])),
            'mse_mean': float(np.mean([m['metrics']['mse_mean'] for m in all_metrics])),
            'mse_std': float(np.mean([m['metrics']['mse_std'] for m in all_metrics])),
            'mse_median': float(np.mean([m['quantiles'].get('mse_q2', m['metrics']['mse_mean']) for m in all_metrics])),
        }
        
        with open(os.path.join(args.save_dir, 'summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        print(f"\nResults for {args.dataset}:")
        print(f"  Pearson: {summary['pearson_mean']:.4f} ± {summary['pearson_std']:.4f}")
        print(f"  MSE: {summary['mse_mean']:.4f}")
        
        return summary
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RetentionMoE Model')
    
    # General
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--datasets', nargs='+', default=['READ'])
    parser.add_argument('--exp_code', type=str, default=None)

    # Model
    parser.add_argument('--model', type=str, default="retention_moe")
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--n_retention_layers', type=int, default=4)
    parser.add_argument('--retention_init_value', type=float, default=2.0)
    parser.add_argument('--retention_heads_range', type=float, default=4.0)
    parser.add_argument('--ffn_ratio', type=float, default=4.0)
    parser.add_argument('--drop_path_rate', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--d_latent', type=int, default=64)
    parser.add_argument('--n_router_heads', type=int, default=4)

    # Data
    parser.add_argument('--source_dataroot', default="/home/zhangyi/code/cma/datasets/hest1k_bench")
    parser.add_argument('--embed_dataroot', type=str, default="/home/zhangyi/code/cma/datasets/hestk1k_embeddings")
    parser.add_argument('--gene_list_dir', default="/home/zhangyi/code/cma/hvg_analysis_select")
    parser.add_argument('--gene_list', type=str, default='final_gene.json')
    parser.add_argument('--save_dir', type=str, default="results/")
    parser.add_argument('--feature_encoder', type=str, default='uni_v2')
    parser.add_argument('--normalize_method', type=str, default="log1p")
    parser.add_argument('--n_genes', type=int, default=50)
    parser.add_argument('--gene_emb_dim', type=int, default=1536)

    # MoE
    parser.add_argument('--protein_embedding_path', type=str, default=None)
    parser.add_argument('--n_macrogenes', type=int, default=64)
    parser.add_argument('--top_k', type=int, default=8)
    parser.add_argument('--cluster_method', type=str, default='kmeans')
    parser.add_argument(
        '--expert_type',
        type=str,
        default='mlp_latent',
        choices=['mlp_latent', 'linear_gene'],
        help='Type of MoE experts: "mlp_latent" (default) or "linear_gene" (per-expert Linear d_model->n_genes).'
    )
    parser.add_argument('--use_gene_group_mask', action='store_true', default=False,
                        help='If set, apply gene-group mask to MoE decoder when protein embeddings are provided. '
                             'Default: False (each expert predicts all genes).')
    parser.add_argument('--save_macrogene_info', action='store_true', default=False)
    parser.add_argument('--global_weight', type=float, default=0.3,
                        help='Fixed fusion weight for global linear head in MoE head. '
                             'final = (1-global_weight)*moe + global_weight*global')
    parser.add_argument('--group_mask_eps', type=float, default=0.0,
                        help='Epsilon value for gene-group mask (when enabled). '
                             '0.0=hard mask, 0.1=soft leakage.')
    parser.add_argument('--group_mask_eps_start', type=float, default=0.8,
                        help='Group-mask eps at early training (soft mask).')
    parser.add_argument('--group_mask_eps_end', type=float, default=0.1,
                        help='Group-mask eps at late training (harder mask).')
    parser.add_argument('--group_mask_eps_decay_epochs', type=int, default=0,
                        help='Number of epochs to decay eps from start to end (0 => use --epochs).')

    # Loss
    parser.add_argument('--mse_weight', type=float, default=1.0)
    parser.add_argument('--pcc_weight', type=float, default=1.0)

    # Training
    parser.add_argument('--clip_norm', type=float, default=1.)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--earlystopping', type=int, default=20)
    parser.add_argument('--save_step', type=int, default=-1)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--sample_times', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use mixed precision training (AMP) to reduce memory')
    parser.add_argument('--no_amp', dest='use_amp', action='store_false',
                        help='Disable mixed precision training')
    parser.add_argument('--attention_chunk_size', type=int, default=None,
                        help='Chunk size for attention to reduce memory. None=full attention, e.g., 512=chunked')

    args = parser.parse_args()
    
    # Feature Dimension
    args.feature_dim = {
        "uni_v1": 1024, "uni_v2": 1536, "gigapath": 1536,
        "virchow2": 2560, "conch_v1": 512,
    }[args.feature_encoder]
    
    args.normalize_method_name = args.normalize_method
    args.normalize_method = get_normalize_method(args.normalize_method)
    set_random_seed(args.seed)

    # Save dir
    base_save_dir = f"results_{args.model}-hest"
    timestamp_dir = get_current_time() if args.exp_code is None else get_current_time() + f"-{args.exp_code}"
    timestamp_dir +=  f"-mse-{args.mse_weight}-pcc-{args.pcc_weight}-seed-({args.seed})---n_macrogenes-({args.n_macrogenes})"
    main_save_dir = os.path.join(args.save_dir, base_save_dir, timestamp_dir)
    os.makedirs(main_save_dir, exist_ok=True)
    print(f"\n>>> Results: {main_save_dir}")

    if args.use_wandb:
        wandb.init(project="rma-hest", name=timestamp_dir, save_code=True)
        wandb.config.update(args)
        wandb.save("models/rma.py")
        wandb.define_metric("epoch") 
        wandb.define_metric("*", step_metric="epoch")

    # Target Datasets
    target_datasets = ["PRAD", "READ", "CCRCC", "HCC", "LYMPH_IDC"]
    if args.datasets[0] == "all":
        args.datasets = target_datasets

    all_results = []
    for dataset in args.datasets:
        print(f"\n{'#'*60}\nProcessing: {dataset}\n{'#'*60}")
        args.dataset = dataset
        args.save_dir = os.path.join(main_save_dir, dataset)
        os.makedirs(args.save_dir, exist_ok=True)

        config_dict = vars(args).copy()
        config_dict.pop('normalize_method', None)
        config_dict['normalize_method'] = args.normalize_method_name
        with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
            json.dump(config_dict, f, indent=4)

        try:
            result = run(args)
            if result:
                all_results.append(result)
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
        torch.cuda.empty_cache()

    # Summary
    if all_results:
        df = pd.DataFrame(all_results)
        columns_order = ['dataset', 'pearson_mean', 'pearson_std', 'mse_mean', 'mse_std']
        df = df[[c for c in columns_order if c in df.columns]]
        df.to_csv(os.path.join(main_save_dir, 'all_summary.csv'), index=False)
        
        overall = {
            'method': args.model,
            'n_retention_layers': args.n_retention_layers,
            'n_heads': args.n_heads,
            'overall_pearson': float(df['pearson_mean'].mean()),
            'overall_mse': float(df['mse_mean'].mean()),
        }
        all_results = {
            'summary': overall,
            'datasets': all_results
        }
        with open(os.path.join(main_save_dir, 'overall.json'), 'w') as f:
            json.dump(all_results, f, indent=4)

       
        
        print(f"\n{'='*60}")
        print(f"Overall: Pearson={overall['overall_pearson']:.4f}, MSE={overall['overall_mse']:.4f}")
        print(f"{'='*60}")

    print("\nCompleted!")
    if args.use_wandb:
        wandb.finish()