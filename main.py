import torch
import time
import tqdm

import fast_rcnn
import datasets
import config
import utils


if __name__ == '__main__':

    train_dataset = datasets.get_train_dataset()
    train_data_loader = datasets.create_train_loader(train_dataset)
    train_dataset = datasets.get_train_dataset()
    train_data_loader = datasets.create_train_loader(train_dataset)
    
    model = fast_rcnn.create_model(num_classes=config.NUM_CLASSES)
    model = model.to(config.DEVICE)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    train_loss_hist = utils.Averager()
    val_loss_hist = utils.Averager()
    save_best_model = utils.SaveBestModel()
    
    train_loss_list = []
    val_loss_list = []
    
    MODEL_NAME = 'FasterRCNN_ResNet50'
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {config.NUM_EPOCHS}")
        
        train_loss_hist.reset()
        val_loss_hist.reset()
    
        start = time.time()
    
        print('Training')
        progress_bar = tqdm.tqdm(train_data_loader, total=len(train_data_loader))
        for i, data in enumerate(progress_bar):
            optimizer.zero_grad()
            images, targets = data
    
            images = list(image.to(config.DEVICE) for image in images)
            targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)

            
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss_list.append(loss_value)
            train_loss_hist.send(loss_value)
            
            losses.backward()
            optimizer.step()
        
            # update the loss value beside the progress bar for each iteration
            progress_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        
        # validation
        print('Validating')
        progress_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
        for i, data in enumerate(progress_bar):
            images, targets = data
    
            images = list(image.to(config.DEVICE) for image in images)
            targets = [{k: v.to(config.DEVICE) for k, v in t.items()} for t in targets]
    
            with torch.no_grad():
                loss_dict = model(images, targets)
    
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            val_loss_list.append(loss_value)
            val_loss_hist.send(loss_value)
    
            progress_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        
        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch+1} validation loss: {val_loss_hist.value:.3f}")   
        
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
    
        utils.save_best_model(
            val_loss_hist.value, epoch, model, optimizer
        )
        utils.save_model(epoch, model, optimizer)
        utils.save_loss_plot(config.OUT_DIR, train_loss, val_loss)
            
        # sleep for 2 seconds after each epoch
        time.sleep(2)