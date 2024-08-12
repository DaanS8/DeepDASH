import torch
import torch.nn as nn
import torch.nn.functional as F
import dsntnn
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

"""
An unofficial implementation of DeepDASH.

Paper citation:
Hall, A., Victor, B., He, Z., Langer, M., Elipot, M., Nibali, A., & Morgan, S. (2021). The detection, tracking, and temporal action localisation of swimmers for automated analysis. Neural Computing and Applications, 33, 1-19. https://doi.org/10.1007/s00521-020-05485-3
Available on: http://homepage.cs.latrobe.edu.au/zhe/files/swimming_paper.pdf

Code implemention by: Daan Seuntjens
"""


class RegionProposalDD(nn.Module):
    """
    Proposal Phase of DeepDASH.
    Given fused frames, output a heatmap of predicted swimmers' heads.
    """
    def __init__(self, n=5, d=3, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n = n
        self.d = d

        self.block1 = nn.Sequential(
            nn.Conv2d(n * d, 16, kernel_size=3, padding=1), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=3, padding=1), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU()
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=32),
        )

        self.block5 = nn.Sequential(
            nn.Conv2d(32, 1, kernel_size=3, padding=1), 
            nn.Sigmoid()
        )

        self.backbone = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5
        )


    def forward(self, x):
        return self.backbone(x)


class RegionProposalResnet(nn.Module):
    """
    Proposal Phase of DeepDASH with ResNet backbone.
    Given fused frames, output a heatmap of predicted swimmers' heads.
    """
    def __init__(self, n=5, d=3, pretrained=True, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.n = n
        self.d = d
        
        if pretrained:
            resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet18 = models.resnet18(weights=None)

        # Early fusion layer
        self.block1 = nn.Sequential(
            nn.Conv2d(n * d, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool
        )

        # Only use first three Resnet layers
        # to end up with same resolution heatmap
        self.block2 = resnet18.layer1
        self.block3 = resnet18.layer2
        self.block4 = resnet18.layer3

        # Final prediction layer
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, padding=1), 
            nn.Sigmoid()
        )

        self.backbone = nn.Sequential(
            self.block1,
            self.block2,
            self.block3,
            self.block4,
            self.block5
        )

    def forward(self, x):
        return self.backbone(x)



class RegionProposalFPN(nn.Module):
    # Based on: https://www.kaggle.com/code/qdpatidar687/fpn-on-resnet18

    def __init__(self, n=5, d=3, pretrained=True):
        super(RegionProposalFPN, self).__init__()

        self.n = n
        self.d = d
        self.num_features_out = 256

        if pretrained:
            resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        else:
            resnet18 = models.resnet18(weights=None)

        # Early fusion
        self.block1 = nn.Sequential(
            nn.Conv2d(n * d, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet18.bn1,
            resnet18.relu,
            resnet18.maxpool
        )
        self.block2 = resnet18.layer1 
        self.block3 = resnet18.layer2 
        self.block4 = resnet18.layer3 
        self.block5 = resnet18.layer4

        # Lateral connections
        self.lateral_c5 = nn.Conv2d(512, self.num_features_out, kernel_size=1)
        self.lateral_c4 = nn.Conv2d(256, self.num_features_out, kernel_size=1)

        # Smoothing
        self.smooth_p4 = nn.Conv2d(self.num_features_out, self.num_features_out, kernel_size=3, padding=1)

        # Final prediction layer
        self.predict = nn.Sequential(
            nn.Conv2d(self.num_features_out, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Bottom to top
        c1 = self.block1(x)
        c2 = self.block2(c1)
        c3 = self.block3(c2)
        c4 = self.block4(c3)
        c5 = self.block5(c4)

        # Top to bottom
        p5 = self.lateral_c5(c5)
        p4 = self.lateral_c4(c4) + F.interpolate(p5, size=c4.shape[2:], mode='nearest')
        p4 = self.smooth_p4(p4)

        # Only use original feature map size
        out = self.predict(p4)

        return out

class RefinementDeepDash(nn.Module):
    """
    Superclass defining the refinement phases possible for DeepDASH.

    Original paper uses the FullRefinement phase.
    """
    def __init__(self, n=5, d=3, *args, **kwargs) -> None:
        """
        All Refinements must include the number of fused channels.
        
        Args:
            param n: Number of fused frames
            param d: Number of channels in each frame
        """
        super().__init__(*args, **kwargs)
        self.n = n
        self.d = d
    
    def forward(self, x, targets=None):
        pass


class LocationRefinement(RefinementDeepDash):
    """
    Given a crop around a (proposal) of a head region, regression for exact head location and neural net for stroke detection.
    """
    def __init__(self, n=5, d=3, _sigma=0.85, *args, **kwargs) -> None:
        
        super().__init__(n, d, *args, **kwargs)
        self._sigma =_sigma
        self.H = 48
        self.W = 80

        self.body1 = nn.Sequential(
            nn.Conv2d(n * d, 16, kernel_size=3, padding=1, dilation=1), 
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU()
        )

        self.body2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=2, dilation=2), 
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU()
        )

        self.body3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=4, dilation=4), 
            nn.GroupNorm(num_groups=16, num_channels=64),
            nn.ReLU()
        )

        self.body4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=8, dilation=8), 
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU()
        )

        self.body = nn.Sequential(
            self.body1,
            self.body2,
            self.body3,
            self.body4
        )

        self.head_regression1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=16, dilation=16), 
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU()
        )

        self.head_regression2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, dilation=1), 
            nn.GroupNorm(num_groups=64, num_channels=256),
            nn.ReLU()
        )

        self.head_regression3 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, padding=1, dilation=1), 
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.ReLU()
        )

        self.head_regression = nn.Sequential(
            self.head_regression1,
            self.head_regression2,
            self.head_regression3
        )
     
    def forward(self, x, targets=None):
        """
        Given crops, get the predicted offset from the center.
        Params:
            x: inputted crops, with expectetd shape (num_crops, H=48, W=80)
            targets: normalised target coordinates. Normalisation: (-1, -1) is top left corner, (1, 1) is bottom right.
        Returns:
            The predicted (absolute) coordinates, different from the center of the swimmer.
            The total refinement loss, a sum of the MSE of the relative coordinate loss, and Jensen-Shannon divergences loss of the produced heatmaps.
        """

        W, H = 80, 48
        assert x.shape[-2] == H and x.shape[-1] == W

        # feature extraction from crops
        x = self.body(x)

        # head soft-argmax to find expected x, y position
        head_heatmap = self.head_regression(x)        
        head_heatmap = dsntnn.flat_softmax(head_heatmap)
        head_location = dsntnn.dsnt(head_heatmap)

        l2 = None
        if targets is not None:
            # Calculate losses
            targets = targets.unsqueeze(1)
            lc = dsntnn.euclidean_losses(head_location, targets)  # coordinate prediction loss
            lh = dsntnn.js_reg_losses(head_heatmap, targets, sigma_t=self._sigma)  # js divergence loss
            l2 = lc.mean() * 0.5 + lh.mean()

        # Unnormalise coordinates, output pixel location
        unnormalized_head_location = head_location.clone()
        unnormalized_head_location[:, :, 0] = (head_location[:, :, 0] + 1) * W / 2
        unnormalized_head_location[:, :, 1] = (head_location[:, :, 1] + 1) * H / 2

        # Dummy stroke detection
        stroke_detection = torch.full((x.shape[0], 1), -1, dtype=torch.float32).to(x.device) 
        out = torch.cat([unnormalized_head_location.squeeze(1), stroke_detection], dim=1)
    
        return out, l2


class FullRefinement(RefinementDeepDash):
    """
    Given a crop around a (proposal) of a head region, regression for exact head location and neural net for stroke detection.
    """
    def __init__(self, n=5, d=3, sigma_t=0.85, *args, **kwargs) -> None:
        super().__init__(n, d, *args, **kwargs)
        
        self.sigma_t = sigma_t

        self.body1 = nn.Sequential(
            nn.Conv2d(n * d, 16, kernel_size=3, padding=1, dilation=1), 
            nn.GroupNorm(num_groups=4, num_channels=16),
            nn.ReLU()
        )

        self.body2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=2, dilation=2), 
            nn.GroupNorm(num_groups=8, num_channels=32),
            nn.ReLU()
        )

        self.body3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=4, dilation=4), 
            nn.GroupNorm(num_groups=16, num_channels=64),
            nn.ReLU()
        )

        self.body4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=8, dilation=8), 
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU()
        )

        self.body = nn.Sequential(
            self.body1,
            self.body2,
            self.body3,
            self.body4
        )

        self.head_regression1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=16, dilation=16), 
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU()
        )

        self.head_regression2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1, dilation=1), 
            nn.GroupNorm(num_groups=64, num_channels=256),
            nn.ReLU()
        )

        self.head_regression3 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, padding=1, dilation=1), 
            nn.GroupNorm(num_groups=1, num_channels=1),
            nn.ReLU()
        )

        self.head_regression = nn.Sequential(
            self.head_regression1,
            self.head_regression2,
            self.head_regression3
        )

        self.stroke_detection1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=16, dilation=16), 
            nn.GroupNorm(num_groups=32, num_channels=128),
            nn.ReLU()
        )

        self.stroke_detection2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2), 
            nn.GroupNorm(num_groups=64, num_channels=256),
            nn.ReLU()
        )

        self.stroke_detection3 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3), 
            nn.AdaptiveMaxPool2d(1),
            nn.Sigmoid()
        )

        self.stroke_detection = nn.Sequential(
            self.stroke_detection1,
            self.stroke_detection2,
            self.stroke_detection3
        )    

     
    def forward(self, x, targets=None):

        # feature extraction from crops
        x = self.body(x)

        # head soft-argmax to find expected x, y position
        head_heatmap = self.head_regression(x)        
        head_heatmap = dsntnn.flat_softmax(head_heatmap)
        head_location = dsntnn.dsnt(head_heatmap)
        
        # stroke detection
        stroke_detection = self.stroke_detection(x)

        combined_loss = None
        if targets is not None:
            target_head_coordinates = x.target[:, :2]
            target_stroke_labels = x.target[:, 2]
            
            euc_losses = dsntnn.mse_losses(head_location, target_head_coordinates)  # regression loss coordinates
            reg_losses = dsntnn.js_reg_losses(head_heatmap, target_head_coordinates, self.sigma_t)  # js divergence loss
            stroke_loss = F.mse_loss(stroke_detection, target_stroke_labels)  # stroke regression loss
            combined_loss = euc_losses + reg_losses + stroke_loss

        out = torch.cat([head_location, stroke_detection], dim=1)
        return out, combined_loss


class DeepDASH(nn.Module):
    """
    The original DeepDASH model
    """
    def __init__(self, region_proposal=RegionProposalDD(n=5, d=3), refinement: RefinementDeepDash = LocationRefinement(), pt=0.2, _gamma=12, _sigma=0.425, _lambda=0.2, *args, **kwargs) -> None:
        """
        The DeepDash model implementation.
        Args:
            param region_proposal: The region proposal phase to use
            param refinement: The refinement stage to use
            param pt: Crop proposal treshold value
            param _gamma (γ): Weight for weighted MSE proposal heatmap
            param _sigma (σ):  Standard deviation STD used in Guassian
            param _lambda (λ): loss weight for refinement
            param crop_size: Size of the crop taken of heatmap in proposal phase (width, height)
        """
        super().__init__(*args, **kwargs)
        
        # The crop size to take on the heatmap (H, W)
        CROP_SIZE=(3, 5)
        
        # Inits
        self.n = refinement.n
        self.d = refinement.d
        self.pt = pt
        self._gamma = _gamma
        self._sigma = _sigma
        self._lambda = _lambda
        self.crop_size = CROP_SIZE
        self.region_proposal = region_proposal
        self.refinement = refinement
        self.device = device

    def nms(self, heatmap):
        """
        Performs non-maximum suppression on a heatmap.
        """
        # Find padding
        w, h = self.crop_size
        padding_w = w // 2
        padding_h = h // 2

        # Perform non-maximum supression: 
        # find largest value in CROP_SIZE field
        heatmap_nms = F.max_pool2d(heatmap, kernel_size=self.crop_size, stride=1, padding=(padding_w, padding_h)) 
        # if max value is different than current value then its not the local maxumum -> set to 0
        heatmap_nms = (heatmap_nms == heatmap).float() * heatmap
        return heatmap_nms.squeeze(1)
    

    def forward(self, inputs, targets_batch=None):
        """
        Performs both inference and training passes.
        """
        # Init params
        FRAME_W, FRAME_H =inputs.shape[-1],  inputs.shape[-2]
        CROP_W, CROP_H = 80, 48   # Image size to input to refinement
        
        # Heatmap construction with non-maximum supression
        head_heatmap = self.region_proposal(inputs)
        head_heatmap_nms = self.nms(head_heatmap)
        head_proposal = head_heatmap_nms > self.pt

        # Construct crops for refinement phase
        head_targets = None
        head_out_base = []
        if targets_batch is not None:  
            # Training mode: crops around targets
            nb_targets = sum([len(targets) for targets in targets_batch])
            head_crops = torch.zeros(size=(nb_targets, self.n * self.d, CROP_H, CROP_W), dtype=torch.float32)
            head_targets = torch.zeros(size=(nb_targets, 2), dtype=torch.float32, device=device)
            i = 0
            for batch_nb, targets in enumerate(targets_batch):
                head_out_base_batch = torch.zeros((len(targets), 2), dtype=torch.float32, device=device)
                for j, box in enumerate(targets):
                    x, y, w, h = box.tolist()
                    cx, cy = x + w/2, y + h/2

                    # 90% Jitter augmentation
                    variance_x = random.uniform(-0.45, 0.45) * CROP_W
                    variance_y = random.uniform(-0.45, 0.45) * CROP_H
                    
                    crop_x = round(max(CROP_W/2 , min(cx + variance_x, FRAME_W - CROP_W/2)))
                    crop_y = round(max(CROP_H/2 , min(cy + variance_y, FRAME_H - CROP_H/2)))

                    x1 = int(crop_x - CROP_W/2)
                    x2 = int(crop_x + CROP_W/2)
                    y1 = int(crop_y - CROP_H/2)
                    y2 = int(crop_y + CROP_H/2)

                    head_crops[i] = inputs[batch_nb, :, y1:y2, x1:x2]

                    head_targets[i] = torch.tensor([-variance_x / CROP_W * 2, -variance_y / CROP_H * 2], device=device, dtype=torch.float32)
                    head_out_base_batch[j] = torch.tensor([crop_x, crop_y], device=device, dtype=torch.float32)
                    i+=1
                head_out_base.append(head_out_base_batch)
        else: 
            # Inference mode: extract crops around heatmap outputs

            # Limit the number of head crops per batch to 100
            for i in range(len(head_proposal)):
                if head_proposal[i].sum() > 100:
                    layer_pt = torch.min(head_heatmap_nms[i].view(-1).topk(101)[0])
                    head_proposal[i] = head_heatmap_nms[i] > layer_pt  # value must be higher than the 101th value

            # Params init
            for i in range(len(head_proposal)):
                head_out_base.append([])
            head_indices = torch.nonzero(head_proposal, as_tuple=False)
            height_halved_crop =  self.crop_size[0] * 8
            width_halved_crop = self.crop_size[1] * 8
            head_crops = torch.zeros(size=(len(head_indices), self.n * self.d, 16*self.crop_size[0], 16*self.crop_size[1]), dtype=torch.float32)

            # Extract crops at predicted locations
            for i, head in enumerate(head_indices):
                assert len(head) == 3
                batch, y_offset_heatmap, x_offset_heatmap = head

                # Translate heatmap coords to image coords
                # Ensure edge cases (borders of image) use valid crops of the image
                y_offset_crop  = max(height_halved_crop, min(FRAME_H - height_halved_crop, y_offset_heatmap * 16))
                x_offset_crop  = max(width_halved_crop, min(FRAME_W - width_halved_crop, x_offset_heatmap * 16))

                x1 = x_offset_crop - width_halved_crop
                x2 = x_offset_crop + width_halved_crop
                y1 = y_offset_crop - height_halved_crop
                y2 = y_offset_crop + height_halved_crop
                head_crops[i] = inputs[batch, :, y1:y2, x1:x2]

                cx, cy = (x1 + x2)/2, (y1 + y2)/2
                head_out_base[batch].append(torch.tensor([cx, cy], device=device, dtype=torch.float32))

            for i in range(len(head_proposal)):
                if len(head_out_base[i]) == 0:
                    head_out_base[i] = torch.empty((0,), device=device, dtype=torch.float32)
                else:
                    head_out_base[i] = torch.stack(head_out_base[i], dim=0).to(device=device, dtype=torch.float32)
        
        # Input crops into refinement stage
        head_crops = head_crops.to(device)
        refinement_out, l2 = self.refinement(head_crops, head_targets)
        
        # Overall loss calculations
        loss = None
        if targets_batch is not None:
            # Param init
            HEATMAP_H, HEATMAP_W = head_heatmap.shape[-2:]
            hg_batch = torch.zeros_like(head_heatmap)

            # Construct gt region proposal heatmap
            for batch_nb, targets in enumerate(targets_batch):
                if len(targets) > 0:
                    normalised_means = torch.zeros(size=(len(targets), 2), dtype=torch.float32)
                    for i, box in enumerate(targets):
                        x, y, w, h = box
                        cx, cy = x + w/2, y + h/2
                        cx /= 16
                        cy /= 16

                        normalised_means[i, 0] = cx / HEATMAP_W * 2 - 1
                        normalised_means[i, 1] = cy / HEATMAP_H * 2 - 1

                    hg_channels = dsntnn.make_gauss(normalised_means, size=(HEATMAP_H, HEATMAP_W), sigma=self._sigma)
                    hg = torch.max(hg_channels, axis=0).values
                else:
                    hg = torch.zeros_like(head_heatmap[0])
                
                hg_batch[batch_nb] = hg.to(device)

            # Weighted MSE of heatmap
            weights = hg_batch * (self._gamma - 1) + 1
            mse = F.mse_loss(hg_batch, head_heatmap, reduction='none')
            l1 = torch.multiply(weights, mse).mean((2, 3)).sum()

            # Total loss = region proposal loss + refinement loss
            loss = l1 + self._lambda * l2

        head_co_out = []
        i = 0
        for base_cos in head_out_base:
            head_co_batch_out = torch.zeros((len(base_cos), 3), dtype=torch.float32, device=device)
            for j, base_co in enumerate(base_cos):
                head_co_batch_out[j, :2] = base_co + refinement_out[i, :2]
                head_co_batch_out[j, 2]  = refinement_out[i, 2]
                i += 1
            head_co_out.append(head_co_batch_out)

        return head_co_out, loss
