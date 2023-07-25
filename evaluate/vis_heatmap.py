import torch
import cv2

def visualize_heatmap_single_sample(img,
                                    pred_heatmap,
                                    gt_heatmap,):
    path = 'tools/analysis/vis_heatmap/'
    
    img_single = img[0]
    pred_heatmap_single = pred_heatmap[0]
    gt_heatmap_single = gt_heatmap[0]
    
    img_h, img_w = img_single.shape[-2:]
    K, heatmap_h, heatmap_w = pred_heatmap_single.shape
    
    pred_heatmap_interp = torch.nn.functional.interpolate(pred_heatmap_single.unsqueeze(0), scale_factor=img_h / heatmap_h, 
                                                                 mode='nearest').squeeze()
    gt_heatmap_interp = torch.nn.functional.interpolate(gt_heatmap_single.unsqueeze(0), scale_factor=img_h / heatmap_h, 
                                                               mode='nearest').squeeze()
    
    
    # pred_heatmap_interp = (pred_heatmap_interp - torch.min(pred_heatmap_interp, dim=0)[0]) / \
    #                         (torch.max(pred_heatmap_interp, dim=0)[0] - torch.min(pred_heatmap_interp, dim=0)[0] + 1e-5)
    # gt_heatmap_interp = (gt_heatmap_interp - torch.min(gt_heatmap_interp, dim=0)[0]) / \
    #                         (torch.max(gt_heatmap_interp, dim=0)[0] - torch.min(gt_heatmap_interp, dim=0)[0] + 1e-5)
    # img = (img - torch.min(img, dim=0)[0]) / (torch.max(img, dim=0)[0] - torch.min(img, dim=0)[0] + 1e-5)
    
    pred_heatmap_interp *= 255.0
    gt_heatmap_interp *= 255.0
    
    # pred_heatmap_interp *= 255.0
    # gt_heatmap_interp *= 255.0
    # img *= 255.0
    
    # img_path = path + 'img.png'
    # cv2.imwrite(img_path, img_single.detach().cpu().numpy())
    
    for i in range(K):
        # pred_heatmap_in_img = img[0] + pred_heatmap_interp[i].unsqueeze(0)
        # gt_heatmap_in_img = img[0] + gt_heatmap_interp[i].unsqueeze(0)
    
        pred_heatmap_path = path + 'pred_heatmap_' + str(i + 1) + '.png'
        gt_heatmap_path = path + 'gt_heatmap_' + str(i + 1) + '.png'
        
        cv2.imwrite(pred_heatmap_path, pred_heatmap_interp[i].detach().cpu().numpy())
        cv2.imwrite(gt_heatmap_path, gt_heatmap_interp[i].detach().cpu().numpy())
        
if __name__ == '__main__':
    img = torch.randn(4, 3, 256, 192)
    pred = torch.randn(4, 17, 64, 48)
    gt = torch.randn(4, 17, 64, 48)
    bs, K, img_h, img_w = img.shape
    heatmap_h, heatmap_w = gt.shape[-2:]
    # pred = torch.nn.functional.interpolate(pred[0].unsqueeze(0), scale_factor=img_h / heatmap_h, 
                                            # mode='bilinear', align_corners=True)

    # visualize_heatmap_single_sample(img, pred, gt)