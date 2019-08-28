import matplotlib.pyplot as plt
import seaborn as sns
from graph_lib import gl
"""
############################################################
############ Filling Database of instances ################
"""

def visualize_attention_matrix(question_tokens, passage_tokens, attention_matrix,
                               image_path):
        """
            Text to visualze attention map for.a given exmaple.
            
            question_tokens: List of tokens of the question
            passage_tokens: List of tokens of the passage
            attention_matrix: len(passage) x len(question) matrix with the probabilities 
        """
        
        f = gl.init_figure()
        ax = f.add_axes([0.1, 0.3, 0.8, 0.5])
        ax_attention_words = f.add_axes([0.1, 0.70, 0.8, 0.15])
        ax_attention_words.axis('off')
        
        
        # add image
        cmap = "binary" #cm.get_cmap('coolwarm', 30)
        i = ax.imshow(attention_matrix, interpolation='nearest', cmap=cmap,vmin=0, vmax=1)

        # add colorbar
        cbaxes = f.add_axes([0.95, 0.3, 0.02, 0.5])
        cbar = f.colorbar(i, cax=cbaxes, orientation='vertical')
        cbar.ax.set_xlabel('Probability', labelpad=6)

        # add labels
        ax.set_yticks(range(len(question_tokens)))
        ax.set_yticklabels(question_tokens)
        
        ax.set_xticks(range(len(passage_tokens)))
        ax.set_xticklabels(passage_tokens, rotation=80)
        
        ax.set_xlabel('Passage')
        ax.set_ylabel('Question')
        
        ###########  GET THE MOST ATTENTION WORDS ########
        Nmax_attention_words = 3
        z = (-attention_matrix).argsort(axis = 1)[:,:]
        
        attentioned_passage_words = []
        for i in range (len(question_tokens)):
            attentioned_passage_words.append([])
            for j in range(Nmax_attention_words):
                attentioned_passage_words[-1].append(passage_tokens[z[i,j]] + "(%.1f%%)"%(attention_matrix[i,z[i,j]]*100))
            attentioned_passage_words[-1] = ", ".join(attentioned_passage_words[-1])

        
        text_correspondance = ""
        for i in range (len(question_tokens)):
            text_correspondance += question_tokens[i] + " ---> " + attentioned_passage_words[i] + "\n"
        
        ax_attention_words.text(0,0,text_correspondance)
#        ax2.yaxis.tick_right()
#        ax2.yaxis.set_label_position("right")
        
        f.show()
#        gl.set_fontSizes(ax = [ax1,ax2], title = 20, xlabel = 15, ylabel = 18, 
#                          legend = 12, xticks = 14, yticks = 14)
        gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0.10)
        
        gl.savefig(image_path,  dpi = 100, sizeInches = [10, 6], close = False, bbox_inches = "tight") 
   
def visualize_similarity_matrix(question_tokens, passage_tokens, attention_matrix,
                               image_path):
        """
            Text to visualze attention map for.a given exmaple.
            
            question_tokens: List of tokens of the question
            passage_tokens: List of tokens of the passage
            attention_matrix: len(passage) x len(question) matrix with the probabilities 
        """
        
        f = gl.init_figure()
        ax = f.add_axes([0.1, 0.3, 0.8, 0.5])
        ax_attention_words = f.add_axes([0.1, 0.70, 0.8, 0.15])
        ax_attention_words.axis('off')
        
        
        # add image
        cmap = "binary" #cm.get_cmap('coolwarm', 30)
        i = ax.imshow(attention_matrix, interpolation='nearest', cmap=cmap)

        # add colorbar
        cbaxes = f.add_axes([0.95, 0.3, 0.02, 0.4])
        cbar = f.colorbar(i, cax=cbaxes, orientation='vertical')
        cbar.ax.set_xlabel('Probability', labelpad=6)

        # add labels
        ax.set_yticks(range(len(question_tokens)))
        ax.set_yticklabels(question_tokens)
        
        ax.set_xticks(range(len(passage_tokens)))
        ax.set_xticklabels(passage_tokens, rotation=80)
        
        ax.set_xlabel('Passage')
        ax.set_ylabel('Question')
        
        ###########  GET THE MOST ATTENTION WORDS ########
        Nmax_attention_words = 3
        z = (-attention_matrix).argsort(axis = 1)[:,:]
        
        attentioned_passage_words = []
        for i in range (len(question_tokens)):
            attentioned_passage_words.append([])
            for j in range(Nmax_attention_words):
                attentioned_passage_words[-1].append(passage_tokens[z[i,j]] + "(%.1f)"%(attention_matrix[i,z[i,j]]*100))
            attentioned_passage_words[-1] = ", ".join(attentioned_passage_words[-1])

        
        text_correspondance = ""
        for i in range (len(question_tokens)):
            text_correspondance += question_tokens[i] + " ---> " + attentioned_passage_words[i] + "\n"
        
        ax_attention_words.text(0,0,text_correspondance)
#        ax2.yaxis.tick_right()
#        ax2.yaxis.set_label_position("right")
        
        f.show()
#        gl.set_fontSizes(ax = [ax1,ax2], title = 20, xlabel = 15, ylabel = 18, 
#                          legend = 12, xticks = 14, yticks = 14)
        gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.20, hspace=0.10)
        
        gl.savefig(image_path,  dpi = 100, sizeInches = [10, 6], close = False, bbox_inches = "tight") 
   
def visualize_solution_matrix(passage_tokens, start_and_end_probs, answer,
                               image_path):
        """
            Text to visualze attention map for.a given exmaple.
            
            question_tokens: List of tokens of the question
            passage_tokens: List of tokens of the passage
            attention_matrix: len(passage) x len(question) matrix with the probabilities 
        """
        
        f = gl.init_figure()
        ax = f.add_axes([0.1, 0.3, 0.8, 0.5])
        ax_attention_words = f.add_axes([0.1, 0.70, 0.8, 0.15])
        ax_attention_words.axis('off')
        
        list_probs = ["Span start probs","Span end probs"]
        # add image
        cmap = "binary" #cm.get_cmap('coolwarm', 30)
        i = ax.imshow(start_and_end_probs, interpolation='nearest', cmap=cmap,vmin=0, vmax=1)

        # add colorbar
#        cbaxes = f.add_axes([0.2, 0.1, 0.6, 0.03])
#        cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
        
        cbaxes = f.add_axes([0.95, 0.3, 0.02, 0.4])
        cbar = f.colorbar(i, cax=cbaxes, orientation='vertical')
        
        cbar.ax.set_xlabel('Probability', labelpad=6)

        # add labels
        ax.set_yticks(range(len(list_probs)))
        ax.set_yticklabels(list_probs)
        
        ax.set_xticks(range(len(passage_tokens)))
        ax.set_xticklabels(passage_tokens, rotation=80)
        
        ax.set_xlabel('Passage')
        
        ###########  GET THE MOST ATTENTION WORDS ########
        Nmax_attention_words = 3
        z = (-start_and_end_probs).argsort(axis = 1)[:,:]
        
        attentioned_passage_words = []
        for i in range (len(list_probs)):
            attentioned_passage_words.append([])
            for j in range(Nmax_attention_words):
                attentioned_passage_words[-1].append(passage_tokens[z[i,j]] + "(%.1f%%)"%(start_and_end_probs[i,z[i,j]]*100))
            attentioned_passage_words[-1] = ", ".join(attentioned_passage_words[-1])
        
        
        text_correspondance = ""
        for i in range (len(list_probs)):
            text_correspondance += list_probs[i] + " ----> " + attentioned_passage_words[i] + "\n"
        text_correspondance += "Most likely answer: " + str(answer)
        ax_attention_words.text(0,0,text_correspondance)
#        ax2.yaxis.tick_right()
#        ax2.yaxis.set_label_position("right")
        
        f.show()
#        gl.set_fontSizes(ax = [ax1,ax2], title = 20, xlabel = 15, ylabel = 18, 
#                          legend = 12, xticks = 14, yticks = 14)
        gl.subplots_adjust(left=.09, bottom=.10, right=.90, top=.95, wspace=.30, hspace=0.10)
        
        gl.savefig(image_path,  dpi = 100, sizeInches = [10, 6], close = False, bbox_inches = "tight") 



def distplot_1D_seaborn(data, ax, labels = ['Distribution of Context Length',"",'Context Length']):
    """ Function that generates a 1D distribution plot from the data
         and saves it to disk
    """
    ax = sns.distplot(data, ax = ax, hist = True, norm_hist = True)
    ax.set_title(labels[0])
    ax.set_ylabel(labels[2])
    ax.set_xlabel(labels[1])
    return ax

"""
############# 2D distribution plots ##############
"""
#question type and 

"""
Plots to be made from the dataset
"""

# Distribution of the loss

# Both start and end at the same time

# Mean loss by:
#         - Question type
#         - Question length
#         - Passage length 
#         - Span Answer (it also lenght of answer and why questions is correlated). 