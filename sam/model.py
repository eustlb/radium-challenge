import torch
import torch.nn as nn
import torch.nn.functional as F


class RadSam(nn.Module):
    def __init__(
        self,
        image_encoder,
        mask_decoder,
        prompt_encoder,
        freeze_img_encoder=False
    ):
        super().__init__()
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        # freeze prompt encoder
        for param in self.prompt_encoder.parameters():
            param.requires_grad = False

        if freeze_img_encoder:
            for param in self.image_encoder.parameters():
                param.requires_grad = False

    def postprocess_masks(self, masks, input_size, original_size):

        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )

        masks = masks[..., : input_size[0], : input_size[1]]

        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks

    def forward(self, img_batch, points):
        """
        points needs to be (coords_torch, labels_torch) with:
        - coords_torch shape [B, 1, 2], dtype torch.float32
        - labels_torch shape [B, 1], dtype torch.float32
        """
        image_embedding = self.image_encoder(img_batch)  # (B, 256, 64, 64)

        # do not compute gradients for prompt encoder
        with torch.no_grad():
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=points,
                boxes=None,
                masks=None,
            )

        b, *_ = img_batch.shape

        batched_masks = []
        batched_iou_preductions = []

        for i in range(b):
            low_res_masks, iou_predictions = self.mask_decoder(
                image_embeddings=image_embedding[i].unsqueeze(0),  # (B, 256, 64, 64)
                image_pe=self.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
                sparse_prompt_embeddings=sparse_embeddings[i].unsqueeze(0),  # (B, 2, 256)
                dense_prompt_embeddings=dense_embeddings[i].unsqueeze(0),  # (B, 256, 64, 64)
                multimask_output=False,
            )
            
            masks = self.postprocess_masks(low_res_masks, (1024, 1024), (512, 512))
            batched_masks.append(masks)
            batched_iou_preductions.append(iou_predictions)

        batched_masks = torch.cat(batched_masks, dim=0)
        batched_iou_preductions = torch.cat(batched_iou_preductions, dim=0)

        return batched_masks, batched_iou_preductions
      


    
