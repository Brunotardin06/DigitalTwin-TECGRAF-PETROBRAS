
import lightning as L
    
class ModelModule(L.LightningModule):
    def __init__(self, cfg, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        
    def training_step(self, batch, batch_idx):
        # x, label = batch
        # x = self.model.prepare(x)
        # y_hat = self.model(x)
        # loss = self.criterion(y_hat, label)
        # self.train_metric.update(y_hat, label)
        # self.log(f'{self.prefix}train_loss', loss.detach().cpu().item(), prog_bar=True, on_epoch=True, on_step = False)
        
        # return loss
        pass
    
       
    def validation_step(self, batch, batch_idx):
        # x, label = batch
        # x = self.model.prepare(x)
        # y_hat = self.model(x)
        # loss = self.criterion(y_hat, label)
        # self.val_metric.update(y_hat, label)
        # self.log(f'{self.prefix}val_loss', loss.detach().cpu().item(), prog_bar=True, on_epoch=True, on_step = False)
        # return loss
        pass
    
    
    def predict_step(self, batch, batch_idx):
        # x, _, _ = batch
        # x = self.model.prepare(x)
        # y_hat = self.model(x)
        # return y_hat
        pass
        
    def configure_optimizers(self):
        # optimizer = self.optimizer_class(self.model.parameters(), **self.optimizer_params)
        # scheduler = self.scheduler_class(optimizer, **self.scheduler_params)
        
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "val_f1_score_1",
        #     },
        # }
        pass
        
