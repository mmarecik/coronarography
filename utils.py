import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

import config


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


def cut_frame(image, max_cut=.2, default_cut=0.05):
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
    # values usually are incorrect ...
    if max_cut < 1:
        max_cut = int(H * max_cut)
    if default_cut < 1:
        default_cut = int(H * default_cut)
    if x > max_cut or y > max_cut: # to adjust
        x = y = default_cut
        w = h = H - default_cut
    if w < (H - max_cut) or h < (H - max_cut): # to adjust
        x = y = default_cut
        w = h = H - default_cut      
    return x, y, w, h


def check_bbox(bboxes, nx, ny, ndx, ndy):
    for bbox in bboxes:
        x, y, dx, dy = bbox
        # check if left and top borders do not overlap with bbox 
        if nx > x: nx = x
        if ny > y: ny = y
        # check if bottom and right borders overlap with bbox
        if dx - nx > ndx: ndx = dx - nx
        if dy - ny > ndy: ndy = dy - ny
    return nx, ny, ndx, ndy
    

def cut_border(image, bboxes):
    # crop borders
    nx, ny, ndx, ndy = cut_frame(image)
    nx, ny, ndx, ndy = check_bbox(bboxes, nx, ny, ndx, ndy)
    image_cropped = image[nx:(nx+ndx), ny:(ny+ndy)]
    
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


def visualize_dataset(dataset, num_images=9, figsize=(10, 10)):
    cols = 3
    rows = int(np.ceil(num_images / cols))

    fig = plt.figure(figsize=figsize)
    for i in range(num_images):
        try:
            img_idx = np.random.randint(0, len(dataset))
            image, target = dataset[img_idx]
            fig.add_subplot(rows, cols, i+1)
            for box_num in range(len(target['boxes'])):
                box = target['boxes'][box_num]
                label = config.CLASSES[target['labels'][box_num]]
                cv2.rectangle(
                    image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                    (255, 0, 0), 2
                )
                cv2.putText(
                    image, str(img_idx), (410, 30), 
                    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255)
                )
            plt.imshow(image)
        except Exception as e:
            print(f"Exception with image of index {img_idx}:", e)
    plt.show()
    

def collate_fn(batch):
    """
    To handle the data loading as different images may have different number 
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))


def save_model(epoch, model, optimizer, file_path='outputs/last_model.pth'):
    """
    Function to save the trained model till current epoch, or whenver called
    """
    torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, file_path)


def save_loss_plot(OUT_DIR, train_loss, val_loss):
    figure_1, train_ax = plt.subplots()
    figure_2, valid_ax = plt.subplots()
    train_ax.plot(train_loss, color='tab:blue')
    train_ax.set_xlabel('iterations')
    train_ax.set_ylabel('train loss')
    valid_ax.plot(val_loss, color='tab:red')
    valid_ax.set_xlabel('iterations')
    valid_ax.set_ylabel('validation loss')
    figure_1.savefig(f"{config.OUT_DIR}/train_loss.png")
    figure_2.savefig(f"{config.OUT_DIR}/valid_loss.png")
    print('SAVING PLOTS COMPLETE...')
    plt.close('all')




