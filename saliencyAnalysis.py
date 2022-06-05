import numpy as np
import os
import random
import PIL.Image
from matplotlib import pylab as plt
from pathlib import Path
from data import DataLoad
import tensorflow as tf
from utils import PackageManager
import logging

PackageManager.install_and_import('saliency')
logging.info("Package installed successfully")
import saliency.core as saliency
logging.info("Package import completed successfully")
class Saliency:

    def __init__(self, model,dl) -> None:
        self.model =  model
        self.dl =  dl
        self.class_idx_str = 'class_idx_str'
    
    def PreprocessImage(self):
        return tf.keras.applications.xception.preprocess_input(self.im)
    
    def call_model_function(self,images, call_model_args=None, expected_keys=None):
        target_class_idx =  call_model_args[self.class_idx_str]
        images = tf.convert_to_tensor(images)
        with tf.GradientTape() as tape:
            if expected_keys==[saliency.base.INPUT_OUTPUT_GRADIENTS]:
                tape.watch(images)
                output_layer = self.model(images)
                output_layer = output_layer[:,target_class_idx]
                gradients = np.array(tape.gradient(output_layer, images))
                return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
            else:
                conv_layer, output_layer = self.model(images)
                gradients = np.array(tape.gradient(output_layer, conv_layer))
                return {saliency.base.CONVOLUTION_LAYER_VALUES: conv_layer,
                        saliency.base.CONVOLUTION_OUTPUT_GRADIENTS: gradients}
    def inference(self):
        idx = np.random.randint(0,self.dl.X_.shape[0]) 
        im = self.dl.X_[idx]
        predictions = self.model(np.array([im]))
        prediction_class = np.argmax(predictions[0])
        call_model_args = {self.class_idx_str: prediction_class}
        pred_prob = np.round(predictions[0,prediction_class].numpy(),2)
        return idx,predictions,prediction_class,call_model_args,pred_prob
    
    def computeSaliency(self):
        data =  {'img': [], 'tumorBorder': [], 'original': [],'predictions': [], 'prediction_class': [],'call_model_args':[],'pred_prob':[],'actual': [],'idx': []}
        vizresult = {'Image': [],'tumorBorder': [], 'VG': [] ,'SmoothGrad': [] ,'IG': [] ,'SmoothGrad': [] ,'XRAI_Full': [] ,'Fast_XRAI' :[] ,'VIG': [], 'GIG': [] ,'Blur_IG':[]}
        mycollection = []
        while(len(data['img']) < 3):
            idx,predictions,prediction_class,call_model_args,pred_prob = self.inference()
            if( prediction_class not in data['prediction_class']):
                data['img'].append(self.dl.X_[idx])
                data['tumorBorder'].append(self.dl.Z[idx])
                data['original'].append(self.dl.A[idx])
                data['predictions'].append(predictions)
                data['prediction_class'].append(prediction_class)
                data['call_model_args'].append(call_model_args)
                data['pred_prob'].append(pred_prob)
                data['actual'].append(self.dl.y[idx])
                data['idx'].append(idx)

                im =  self.dl.X_[idx]
                vizresult['Image'].append(self.dl.A[idx])
                vizresult['tumorBorder'].append(self.dl.Z[idx])


                # Construct the saliency object. This alone doesn't do anthing.
                gradient_saliency = saliency.GradientSaliency()

                # Compute the vanilla mask and the smoothed mask.
                vanilla_mask_3d = gradient_saliency.GetMask(im, self.call_model_function, call_model_args)
                smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(im, self.call_model_function, call_model_args)

                # Call the visualization methods to convert the 3D tensors to 2D grayscale.
                vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_mask_3d)
                smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_mask_3d)
                vizresult['VG'].append(vanilla_mask_grayscale)
                vizresult['SmoothGrad'].append(smoothgrad_mask_grayscale)


                mycollection.append(self.dl.A[idx])
                mycollection.append(vanilla_mask_grayscale)
                mycollection.append(smoothgrad_mask_grayscale)


                # Construct the saliency object. This alone doesn't do anthing.
                integrated_gradients = saliency.IntegratedGradients()

                # Baseline is a black image.
                baseline = np.zeros(im.shape)

                # Compute the vanilla mask and the smoothed mask.
                vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
                im, self.call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)
                # Smoothed mask for integrated gradients will take a while since we are doing nsamples * nsamples computations.
                smoothgrad_integrated_gradients_mask_3d = integrated_gradients.GetSmoothedMask(
                im, self.call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)

                # Call the visualization methods to convert the 3D tensors to 2D grayscale.
                vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
                smoothgrad_mask_grayscale = saliency.VisualizeImageGrayscale(smoothgrad_integrated_gradients_mask_3d)

                mycollection.append(vanilla_mask_grayscale)
                mycollection.append(smoothgrad_mask_grayscale)


                # Construct the saliency object. This alone doesn't do anthing.
                xrai_object = saliency.XRAI()

                # Compute XRAI attributions with default parameters
                xrai_attributions = xrai_object.GetMask(im, self.call_model_function, call_model_args, batch_size=20)

                # Show most salient 30% of the image
                mask = xrai_attributions > np.percentile(xrai_attributions, 70)
                im_mask =  np.array(im)
                im_mask[~mask] = 0

                mycollection.append(xrai_attributions)
                mycollection.append(im_mask)

                # Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
                integrated_gradients = saliency.IntegratedGradients()
                guided_ig = saliency.GuidedIG()

                # Baseline is a black image for vanilla integrated gradients.
                baseline = np.zeros(im.shape)

                # Compute the vanilla mask and the Guided IG mask.
                vanilla_integrated_gradients_mask_3d = integrated_gradients.GetMask(
                im, self.call_model_function, call_model_args, x_steps=25, x_baseline=baseline, batch_size=20)
                guided_ig_mask_3d = guided_ig.GetMask(
                im, self.call_model_function, call_model_args, x_steps=25, x_baseline=baseline, max_dist=1.0, fraction=0.5)

                # Call the visualization methods to convert the 3D tensors to 2D grayscale.
                vanilla_mask_grayscale = saliency.VisualizeImageGrayscale(vanilla_integrated_gradients_mask_3d)
                guided_ig_mask_grayscale = saliency.VisualizeImageGrayscale(guided_ig_mask_3d)

                mycollection.append(vanilla_mask_grayscale)
                mycollection.append(guided_ig_mask_grayscale)


                # Compare BlurIG and Smoothgrad with BlurIG. Note: This will take a long time to run.

                # Construct the saliency object. This alone doesn't do anthing.
                blur_ig = saliency.BlurIG()

                # Compute the Blur IG mask and Smoothgrad+BlurIG mask.
                blur_ig_mask_3d = blur_ig.GetMask(im,self.call_model_function, call_model_args, batch_size=20)
                # Smoothed mask for BlurIG will take a while since we are doing nsamples * nsamples computations.
                smooth_blur_ig_mask_3d = blur_ig.GetSmoothedMask(im, self.call_model_function, call_model_args, batch_size=20)

                # Call the visualization methods to convert the 3D tensors to 2D grayscale.
                blur_ig_mask_grayscale = saliency.VisualizeImageGrayscale(blur_ig_mask_3d)
                smooth_blur_ig_mask_grayscale = saliency.VisualizeImageGrayscale(smooth_blur_ig_mask_3d)

                mycollection.append(blur_ig_mask_grayscale)
                mycollection.append(smooth_blur_ig_mask_grayscale)
        return data, vizresult,mycollection
    

    def visualize(self)->None:
        data, vizresult,mycollection = self.computeSaliency()
        titles = ['Input Image', 'Vanilla Gradient','SmoothGrad','Integrated Gradients','SmoothGrad','XRAI Full','Fast XRAI 30%','VIG', 'Guided IG','Blur IG','Smooth Blur IG']
        r,c =  3,len(mycollection)//3 #r is the  number of rows and c is the number of columns
        img_idx =  list(np.arange(0,int(r*c),c))
        fig, axs = plt.subplots(r,c, figsize=(22,6), facecolor='w', edgecolor='k')
        fig.subplots_adjust(hspace = .002, wspace=.001)
        axs = axs.ravel()
        for i in range(int(r*c)):
            if( i in list(range(c))):
                axs[i].set_title(titles[i],fontsize=12)
            axs[i].imshow(np.squeeze(mycollection[i]))
            # axs[i].axis('off')
            if(i  in img_idx):
                if(i == img_idx[0]):
                    tumorBorder = data['tumorBorder'][0]
                    axs[i].set_ylabel(f'Actual: {str(self.dl.CLASSES[dl.y[data["idx"][0]]]).title()}',fontsize=12)
                    axs[i].set_title(f'Prediction: {self.dl.CLASSES[data["prediction_class"][0]].title()}',fontsize=12)
                elif(i == img_idx[1]):
                    tumorBorder = data['tumorBorder'][1]
                    axs[i].set_ylabel(f'Actual: {str(self.dl.CLASSES[dl.y[data["idx"][1]]]).title()}',fontsize=12)
                    axs[i].set_title(f'Prediction: {self.dl.CLASSES[data["prediction_class"][1]].title()}',fontsize=12)
                elif(i == img_idx[2]):
                    tumorBorder = data['tumorBorder'][2]
                    axs[i].set_ylabel(f'Actual: {str(self.dl.CLASSES[dl.y[data["idx"][2]]]).title()}',fontsize=12)
                    axs[i].set_title(f'Prediction: {self.dl.CLASSES[data["prediction_class"][2]].title()}',fontsize=12)
                for j in range(tumorBorder.shape[0]-1):
                    axs[i].scatter(tumorBorder[j],  tumorBorder[j+1],marker=".", color="red", s=200, alpha=0.6,zorder=2)
        fig.tight_layout()
        plt.savefig(f'./Figures/Saliency_Maps_{self.model.name}.svg', bbox_inches ="tight", dpi=300)
        plt.show()

if __name__ == "__main__":
    """
    This routine performs saliency analysis using the trained models saved in the ./Models directory and the dataset
    which is loaded using the DataLoad class
    """
    logging.basicConfig(level=logging.INFO)

    dl =  DataLoad()
    # dl.build(flag=True)
    dl.load()

    model = tf.keras.models.load_model("./Models/xception")

    #We are computing and visualizing the saliency maps in this section
    sa = Saliency(model,dl)

    print(sa.inference())


