import torch
from pytorch_utils.module import LightningModule
from models.common import get_linear_scheduler
from models.autoencoder import AutoEncoder
from evaluation import EMD_CD

class AutoEncoderModule(LightningModule):
    def __init__(self, opt=None, **kwargs) -> None:
        super().__init__(opt=opt, **kwargs)
        self.model = AutoEncoder(self.opt)
        
    def predict_step(self, batch, batch_idx):
        ref = batch['pointcloud']
        shift = batch['shift']
        scale = batch['scale']

        code = self.model.encode(ref)
        recons = self.model.decode(code, ref.size(1), flexibility=self.opt.flexibility)

        ref = ref * scale + shift
        recons = recons * scale + shift
        
        return ref, recons

    
    def forward(self, batch, batch_idx, split):
        x = batch['pointcloud']
        B = x.shape[0]
        loss = self.model.get_loss(x)
        self.log_value('loss', loss, split, B)
        if split == "train": 
           return loss
        
        #ref = batch['pointcloud']
        #shift = batch['shift']
        #scale = batch['scale']
        #with torch.no_grad():
        #    code = self.model.encode(ref)
        #    recons = self.model.decode(code, ref.size(1), flexibility=self.opt.flexibility)
        #refs   = ref    * scale + shift
        #recons = recons * scale + shift
        #
        #metrics = EMD_CD(recons, refs, batch_size=self.opt.val_batch_size)
        #cd, emd = metrics['MMD-CD'].item(), metrics['MMD-EMD'].item()
        #self.log_value('chamfer_distance', cd, split, B)
        #self.log_value('emd', emd, split, B)
        
        return {
            'loss': loss,
            #'chamfer_distance':cd,
            #'emd': emd
            }
    
    
    def configure_optimizers(self):
        # Optimizer and scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), 
            lr=self.opt.learning_rate, 
            weight_decay=self.opt.weight_decay
        )
        scheduler = get_linear_scheduler(
            optimizer,
            start_epoch=self.opt.sched_start_epoch,
            end_epoch=self.opt.sched_end_epoch,
            start_lr=self.opt.learning_rate,
            end_lr=self.opt.end_lr
        )
        
        return [optimizer], [scheduler]