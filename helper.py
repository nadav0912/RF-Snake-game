from IPython import display
import matplotlib.pyplot as plt

plt.ion()

# Create the window with 2 rows 1 cols
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), constrained_layout=True)
fig.canvas.manager.set_window_title('Training Dashboard')
fig.canvas.manager.window.wm_geometry("+0+100")
plt.show(block=False)

def plot(scores, mean_scores, losses):
    # ==========================================
    # Score & Moving Average
    # ==========================================
    ax1.clear()
    ax1.set_title('Training Progress')
    ax1.set_ylabel('Score')
    
    ax1.plot(scores, label='Score', color='blue')
    ax1.plot(mean_scores, label='Moving Average (50)', color='orange', linewidth=2)
    ax1.set_ylim(ymin=0)
    
    if len(scores) > 0:
        ax1.text(len(scores)-1, scores[-1], str(scores[-1]))
        ax1.text(len(mean_scores)-1, mean_scores[-1], str(round(mean_scores[-1], 2)))
        
    ax1.legend(loc='upper left')

    # ==========================================
    # Plot Loss
    # ==========================================
    ax2.clear()
    ax2.set_xlabel('Number of Games')
    ax2.set_ylabel('Loss')
    
    ax2.plot(losses, label='Loss (Long Memory)', color='red')
    ax2.set_ylim(ymin=0)
    
    if len(losses) > 0:
        ax2.text(len(losses)-1, losses[-1], str(round(losses[-1], 4)))
        
    ax2.legend(loc='upper right')

    plt.tight_layout() 
    
    fig.canvas.draw()
    fig.canvas.flush_events()