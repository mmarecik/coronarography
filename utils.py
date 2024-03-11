# import albumentations as A
import cv2
# import numpy as np
# import torch
# import matplotlib.pyplot as plt
# from albumentations.pytorch import ToTensorV2
# from config import DEVICE, CLASSES

# plt.style.use('ggplot')


# this class keeps track of the training and validation loss values...
# ... and helps to get the average for each epoch as well
class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0
        
    def send(self, value):
        self.current_total += value
        self.iterations += 1
    
    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations
    
    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


class SaveBestModel:
    """ Saves the best model.
    
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': current_valid_loss,
                }, 'outputs/best_model.pth')


def cut_frame(image, max_cut=.2, default_cut=60):
    """
    Cuts black frame of the image. (from Martyna)
    """
    # treshold implementation
    [H, W, n] = image.shape
    ret, thresh = cv2.threshold(image, 10, 255, cv2.THRESH_BINARY_INV)
    edged = cv2.Canny(thresh, 0,50)
    contours,_ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
    [x, y, w, h] = sorted(boundRect, key=lambda coef: coef[3])[-1]
    
    # check output points, if points are incorrect set default values
    if max_cut < 1:
        max_cut = H * 0.2
    if x > max_cut or y > max_cut: # to adjust
        x = y = default_cut
        w = h = H - default_cut
    if w < (H - max_cut) or h < (H - max_cut): # to adjust
        x = y = default_cut
        w = h = H - default_cut      
    return x, y, w, h


def cut_border(image, bboxes):
    # crop borders
    nx, ny, ndx, ndy = cut_frame(image)
    image_cropped = image[nx:ndx, ny:ndy]
    
    # calculate new coordinates of bboxes
    bboxes_cropped = []
    for bbox in bboxes:
        x, y, dx, dy = bbox
        bbox_nx, bbox_ny = x - nx, y - ny
        bbox_ndx, bbox_ndy = dx - nx, dy - ny
        bboxes_cropped.append([bbox_nx, bbox_ny, bbox_ndx, bbox_ndy])

    return image_cropped, bboxes_cropped


def custom_transforms(image, bboxes, target_size):
    H, W, _ = image.shape
    H_scaled, W_scaled = target_size /  H, target_size / W

    # resize image and bboxes
    image_resized = cv2.resize(image, (target_size, target_size))
    bboxes_resized = []
    for bbox in bboxes:
        x, y, dx, dy = bbox
        nx = int(np.round(x * W_scaled))
        ny = int(np.round(y * H_scaled))
        ndx = int(np.round(dx * W_scaled))
        ndy = int(np.round(dy * H_scaled))
        bboxes_resized.append([nx, ny, ndx, ndy])

    # cut black borders around image
    image_cropped, bboxes_cropped = cut_border(image_resized, bboxes_resized)
    
    return image_cropped, bboxes_cropped


# TO BE IMPLEMENTED ...

# def collate_fn(batch):
#     """
#     To handle the data loading as different images may have different number 
#     of objects and to handle varying size tensors as well.
#     """
#     return tuple(zip(*batch))
# # define the training tranforms
# def get_train_transform():
#     return A.Compose([
#         A.Flip(0.5),
#         A.RandomRotate90(0.5),
#         A.MotionBlur(p=0.2),
#         A.MedianBlur(blur_limit=3, p=0.1),
#         A.Blur(blur_limit=3, p=0.1),
#         ToTensorV2(p=1.0),
#     ], bbox_params={
#         'format': 'pascal_voc',
#         'label_fields': ['labels']
#     })
# # define the validation transforms
# def get_valid_transform():
#     return A.Compose([
#         ToTensorV2(p=1.0),
#     ], bbox_params={
#         'format': 'pascal_voc', 
#         'label_fields': ['labels']
#     })
# def show_tranformed_image(train_loader):
#     """
#     This function shows the transformed images from the `train_loader`.
#     Helps to check whether the tranformed images along with the corresponding
#     labels are correct or not.
#     Only runs if `VISUALIZE_TRANSFORMED_IMAGES = True` in config.py.
#     """
#     if len(train_loader) > 0:
#         for i in range(1):
#             images, targets = next(iter(train_loader))
#             images = list(image.to(DEVICE) for image in images)
#             targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
#             boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
#             labels = targets[i]['labels'].cpu().numpy().astype(np.int32)
#             sample = images[i].permute(1, 2, 0).cpu().numpy()
#             for box_num, box in enumerate(boxes):
#                 cv2.rectangle(sample,
#                             (box[0], box[1]),
#                             (box[2], box[3]),
#                             (0, 0, 255), 2)
#                 cv2.putText(sample, CLASSES[labels[box_num]], 
#                             (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 
#                             1.0, (0, 0, 255), 2)
#             cv2.imshow('Transformed image', sample)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
# def save_model(epoch, model, optimizer):
#     """
#     Function to save the trained model till current epoch, or whenver called
#     """
#     torch.save({
#                 'epoch': epoch+1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 }, 'outputs/last_model.pth')
# def save_loss_plot(OUT_DIR, train_loss, val_loss):
#     figure_1, train_ax = plt.subplots()
#     figure_2, valid_ax = plt.subplots()
#     train_ax.plot(train_loss, color='tab:blue')
#     train_ax.set_xlabel('iterations')
#     train_ax.set_ylabel('train loss')
#     valid_ax.plot(val_loss, color='tab:red')
#     valid_ax.set_xlabel('iterations')
#     valid_ax.set_ylabel('validation loss')
#     figure_1.savefig(f"{OUT_DIR}/train_loss.png")
#     figure_2.savefig(f"{OUT_DIR}/valid_loss.png")
#     print('SAVING PLOTS COMPLETE...')
#     plt.close('all')





