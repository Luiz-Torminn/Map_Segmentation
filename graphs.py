#%%
from matplotlib import pyplot as plt

#%%
def loss_graph(loss_dict, fig_title = None, multiple_graphs = None, save_path = None):
    
    fig = plt.figure(figsize=(10,5))
    
    if not multiple_graphs:
        ax = fig.add_subplot()
        ax = fig.gca()
    
        for (k, v) in loss_dict.items():   
            ax.plot([n for n in range(len(v))], v, label = f"{k.title()}")
        
        ax.legend()
    
    if multiple_graphs:
        for i, (k, v) in enumerate(loss_dict.items()):   
            ax = fig.add_subplot(1, len(loss_dict.keys()), (1 + i))
            ax = fig.gca()
            
            ax.plot([n for n in range(len(v))], v, label = f"{k.title()}")
            ax.set_title(f"{k.title()}")
            ax.legend()
            
        
    if fig_title:
        fig.suptitle(f"{fig_title.title()}")
        
    plt.subplots_adjust(wspace=5)
    plt.tight_layout()
        
            
    if save_path:
        plt.savefig(save_path)

    plt.show()

#%%
def plot_images(original_image, original_mask, prediction, save_path = None):
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12,5))
    ax1.imshow(original_image)
    ax2.imshow(original_mask)
    ax3.imshow(prediction)

    ax1.set_title("Original Image")
    ax2.set_title("Original Mask")
    ax3.set_title("Prediction")

    plt.suptitle("IMAGES", fontweight = "bold")
    plt.subplots_adjust(wspace=.5)
    
    if save_path:
        plt.savefig(save_path)
        
    plt.show()

    
#%%
# loss_hist = {
#     "Train Loss": [10, 12, 20, 34],
#     "Validation Loss": [21, 12, 8, 5],
# }

# loss_graph(loss_hist, "Loss Values", multiple_graphs=False)
    
# %%
