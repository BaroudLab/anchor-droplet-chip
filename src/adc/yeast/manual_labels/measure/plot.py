import matplotlib.pyplot as plt
import seaborn as sns


#function to plot tracked labels over time 
def plot_tracked_labels(df, title=""):
    #calculate the ratio column by dividing top10px by mean_intensity 
    df.loc[:, "ratio"] = df.top10px / df.mean_intensity

    #creat a 2x1 subplot layout
    fig, ax = plt.subplots(ncols=2, figsize = (10,6))
    #conver the 'label' column to categorical for better visualization 
    df.label = df.label.astype("category")
    #create a line plot for top10px over hours with specific settings 
    sns.lineplot(
        ax=ax[0],
        data=df,
        x='hours',
        y='top10px',
        hue='label',
        palette='Set2',
        style="channel",
        legend=False
    )
    sns.lineplot(
        ax=ax[0],
        data=df,
        x='hours',
        y='cyto_wo100px',
        hue='label',
        palette='Set2',
        style="channel",
        legend=False
    )
    ax[0].set_title(title)
    #create a line plot for ratio over hours with specific settings
    sns.lineplot(ax=ax[1],
             data=df,
             x='hours',
             y='ratio',
             hue='label',
             palette='Set2',
             style="channel"
            )
    ax[1].legend(loc=(1.1,0)) #adjust legened location 
    ax[1].set_title(title)
    
    plt.tight_layout()
    return fig

#function to plot tracked labels over time for GFP data 
def plot_tracked_labels_gfp(df, title=""):
    #calculate ratio column by dividing top10px by mean intensity 
    df.loc[:, "ratio"] = df.top10px / df.mean_intensity
    #create 2x1 subplot layout
    fig, ax = plt.subplots(ncols=2, figsize = (10,6))
    #convert 'label' column to categorical for better visualization 
    df.label = df.label.astype("category")
    #create a line plot for top10px over GFPhour with specific settings 
    sns.lineplot(
        ax=ax[0],
        data=df,
        x='GFPhour',
        y='top10px',
        hue='label',
        palette='Set2',
        style="channel",
        legend=False
    )
    ax[0].set_title(title)
    #create a line plot for ratio over GFP hours with specific settings 
    sns.lineplot(ax=ax[1],
             data=df,
             x='GFPhour',
             y='ratio',
             hue='label',
             palette='Set2',
             style="channel"
            )
    ax[1].legend(loc=(1.1,0))
    ax[1].set_title(title)
    plt.tight_layout()
    return fig
