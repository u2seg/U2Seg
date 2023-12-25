from modules import *
from data import *
from collections import defaultdict
from multiprocessing import Pool
import hydra
import seaborn as sns
import torch.multiprocessing
from crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_segmentation import LitUnsupervisedSegmenter, prep_for_plot, get_class_labels
import pdb
import time

torch.multiprocessing.set_sharing_strategy('file_system')

def plot_cm(histogram, label_cmap, cfg):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    hist = histogram.detach().cpu().to(torch.float32)
    hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
    sns.heatmap(hist.t(), annot=False, fmt='g', ax=ax, cmap="Blues", cbar=False)
    ax.set_title('Predicted labels', fontsize=28)
    ax.set_ylabel('True labels', fontsize=28)
    names = get_class_labels(cfg.dataset_name)
    if cfg.extra_clusters:
        names = names + ["Extra"]
    ax.set_xticks(np.arange(0, len(names)) + .5)
    ax.set_yticks(np.arange(0, len(names)) + .5)
    ax.xaxis.tick_top()
    ax.xaxis.set_ticklabels(names, fontsize=18)
    ax.yaxis.set_ticklabels(names, fontsize=18)
    colors = [label_cmap[i] / 255.0 for i in range(len(names))]
    [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
    [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.vlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_xlim())
    ax.hlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_ylim())
    plt.tight_layout()


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])


def batched_crf(pool, img_tensor, prob_tensor):
    outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)


@hydra.main(config_path="configs", config_name="batch_infer_config_imagenet.yml")
def my_app(cfg: DictConfig) -> None:
    pytorch_data_dir = cfg.pytorch_data_dir
    result_dir = "../results/predictions/{}".format(cfg.experiment_name)
    os.makedirs(join(result_dir, "img"), exist_ok=True)
    os.makedirs(join(result_dir, "label"), exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)
    os.makedirs(join(result_dir, "picie"), exist_ok=True)

    for model_path in cfg.model_paths:
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        print(OmegaConf.to_yaml(model.cfg))

        run_picie = cfg.run_picie and model.cfg.dataset_name == "cocostuff27"
        if run_picie:
            picie_state = torch.load("../saved_models/picie_and_probes.pth")
            picie = picie_state["model"].cuda()
            picie_cluster_probe = picie_state["cluster_probe"].module.cuda()
            picie_cluster_metrics = picie_state["cluster_metrics"]

        loader_crop = "center"
        # test_dataset = ContrastiveSegDataset(
        #     pytorch_data_dir=pytorch_data_dir,
        #     dataset_name=model.cfg.dataset_name,
        #     crop_type=None,
        #     image_set="val",
        #     transform=get_transform(cfg.res, False, loader_crop),
        #     target_transform=get_transform(cfg.res, True, loader_crop),
        #     cfg=model.cfg,
        # )
        
        test_dataset = ContrastiveSegDataset(
            pytorch_data_dir=pytorch_data_dir,
            dataset_name="imagenet",
            crop_type=None,
            image_set="val",
            transform=get_transform(cfg.res, False, loader_crop),
            target_transform=get_transform(cfg.res, True, loader_crop),
            cfg=model.cfg,
        )

        test_loader = DataLoader(test_dataset, cfg.batch_size * 2,
                                 shuffle=False, num_workers=cfg.num_workers,
                                 pin_memory=True, collate_fn=flexible_collate)

        model.eval().cuda()
        if cfg.use_ddp:
            par_model = torch.nn.DataParallel(model.net)
            if run_picie:
                par_picie = torch.nn.DataParallel(picie)
        else:
            par_model = model.net
            if run_picie:
                par_picie = picie

        if model.cfg.dataset_name == "cocostuff27":
            # all_good_images = range(10)
            # all_good_images = range(250)
            # all_good_images = [61, 60, 49, 44, 13, 70] #Failure cases
            all_good_images = [19, 54, 67, 66, 65, 75, 77, 76, 124]  # Main figure
        elif model.cfg.dataset_name == "cityscapes":
            # all_good_images = range(80)
            # all_good_images = [ 5, 20, 56]
            all_good_images = [11, 32, 43, 52]
        else:
            raise ValueError("Unknown Dataset {}".format(model.cfg.dataset_name))
        batch_nums = torch.tensor([n // (cfg.batch_size * 2) for n in all_good_images])
        batch_offsets = torch.tensor([n % (cfg.batch_size * 2) for n in all_good_images])

        # Ensure the save directory exists
        save_dir = cfg.output_root
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        saved_data = defaultdict(list)
        # with Pool(cfg.num_workers) as pool:
        from multiprocessing import get_context
        with get_context('spawn').Pool(cfg.num_workers + 5) as pool:

            for i, batch in enumerate(tqdm(test_loader)):
                start_time = time.time()  

                if (torch.max(batch['ind']) < 1100000).item():
                    print("skip:", torch.max(batch['ind']).item())
                    continue                

                with torch.no_grad():
                    img = batch["img"].cuda()
                    label = batch["label"].cuda()

                    predict_start = time.time()
                    feats, code1 = par_model(img)
                    feats, code2 = par_model(img.flip(dims=[3]))
                    code = (code1 + code2.flip(dims=[3])) / 2
                    code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)
                    cluster_probs = model.cluster_probe(code, 2, log_probs=True)
                    predict_end = time.time() 
                    
                    if cfg.run_crf:
                        crf_start = time.time()  
                        cluster_preds = batched_crf(pool, img, cluster_probs).argmax(1).cuda()
                        crf_end = time.time()  
                    else:
                        cluster_preds = cluster_probs.argmax(1)

                transfer_start = time.time()
                cluster_preds_cpu = cluster_preds.cpu().numpy().astype(np.int32)
                transfer_end = time.time()
                
                save_start = time.time()
                for idx, ind in enumerate(batch['ind']):
                    file_path = os.path.join(save_dir, f"{ind.item()}.npy")
                    np_array = cluster_preds_cpu[idx]
                    np.save(file_path, np_array)
                save_end = time.time()

                print(f"Batch {i} - Model Prediction Time: {predict_end - predict_start:.4f} seconds")
                if cfg.run_crf:
                    print(f"Batch {i} - CRF Processing Time: {crf_end - crf_start:.4f} seconds")
                print(f"Batch {i} - Data Transfer Time: {transfer_end - transfer_start:.4f} seconds")
                print(f"Batch {i} - Save Time: {save_end - save_start:.4f} seconds")
                print(f"Batch {i} - Total Time: {save_end - start_time:.4f} seconds")
                print("-" * 50)



if __name__ == "__main__":
    prep_args()
    my_app()
