import matplotlib.pyplot as plt

colors = {'R': 'red', 'G': 'green', 'B': 'blue'}

def get_dico_values(ds,data):

    L_images=[]

    for id in range(100000):
        L_images.append(ds[id][0])


    # Initialize lists to store min, max, mean values for each channel
    min_values = {'R': [], 'G': [], 'B': []}
    max_values = {'R': [], 'G': [], 'B': []}
    mean_values = {'R': [], 'G': [], 'B': []}
    std_values= {'R': [], 'G': [], 'B': []}



    dico_stats_labels=dict()

    # Process each image
    for id,img in tqdm(enumerate(L_images)):

        labels=get_labels(data,id)
        for i, channel in enumerate(['R', 'G', 'B']):
            min_values[channel].append(img[i].min().item())
            max_values[channel].append(img[i].max().item())
            mean_values[channel].append(img[i].mean().item())
            std_values[channel].append(img[i].std().item())

        for tmp_labels in labels:
            if not tmp_labels in dico_stats_labels:
                dico_stats_labels[tmp_labels]=dict()

                dico_stats_labels[tmp_labels]["min_values"]={'R': [], 'G': [], 'B': []}
                dico_stats_labels[tmp_labels]["max_values"]={'R': [], 'G': [], 'B': []}
                dico_stats_labels[tmp_labels]["mean_values"]={'R': [], 'G': [], 'B': []}
                dico_stats_labels[tmp_labels]["std_values"]={'R': [], 'G': [], 'B': []}

            for i, channel in enumerate(['R', 'G', 'B']):
                dico_stats_labels[tmp_labels]["min_values"][channel].append(img[i].min().item())
                dico_stats_labels[tmp_labels]["max_values"][channel].append(img[i].max().item())
                dico_stats_labels[tmp_labels]["mean_values"][channel].append(img[i].mean().item())
                dico_stats_labels[tmp_labels]["std_values"][channel].append(img[i].std().item())
    
    dico_stats_labels["SUMMARY"]=dict()
    dico_stats_labels["SUMMARY"]["min_values"]=min_values
    dico_stats_labels["SUMMARY"]["max_values"]=max_values
    dico_stats_labels["SUMMARY"]["mean_values"]=mean_values
    dico_stats_labels["SUMMARY"]["std_values"]=std_values

    return dico_stats_labels



# Function to plot histograms
def plot_histogram(data, title, xlabel, ylabel):
    plt.figure()
    for channel, values in data.items():
        plt.hist(values, bins=10, alpha=0.5, label=channel,color=colors[channel])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()

def plot_histograms_side_by_side(data_dict, titles, xlabel, ylabel, suptitle=None):
    channels = ["R", "G", "B"]  # Assuming fixed channels
    num_plots = len(data_dict)  # Number of histogram types (min, max, mean, std)
    
    plt.figure(figsize=(5 * num_plots, 4))  # Adjust figure size based on the number of plots
    
    for i, (stat, channel_data) in enumerate(data_dict.items()):
        plt.subplot(1, num_plots, i + 1)
        
        for channel in channels:
            if channel in channel_data:
                plt.hist(channel_data[channel], bins=10, alpha=0.5, label=channel, color=colors[channel])
        
        plt.title(titles[i])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
    
    if suptitle:  # Add a big title if provided
        plt.suptitle(suptitle, fontsize=16, y=1.02)  # Adjust the fontsize and vertical position
    
    plt.tight_layout()
    
    if suptitle:  # Save the plot to a PDF if a suptitle is provided
        filename = f"./plots/{suptitle.replace(' ', '_')}_histos.pdf"  # Replace spaces with underscores
        plt.savefig(filename, format='pdf', bbox_inches='tight')  # Save the figure as a PDF
    
    plt.show()

def get_dico_summary(dico_stats_labels):
    for key in dico_stats_labels:

        plot_histograms_side_by_side(
            dico_stats_labels[key],
            titles=['Histogram of Min Values', 'Histogram of Max Values', 'Histogram of Mean Values', 'Histogram of Std Values'],
            xlabel='Value',
            ylabel='Frequency',
            suptitle=key)


