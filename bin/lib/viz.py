# -*- coding: utf-8 -*-
import sys, re, itertools, copy, logging
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
from keras.utils import plot_model
from keras import activations
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, precision_score, f1_score, recall_score
from scipy import interp
import boto3, tempfile
from os.path import basename
from datetime import datetime as dt
import matplotlib.dates as mdates
from matplotlib.dates import MO, TU, WE, TH, FR, SA, SU


class Viz:

    s3 = False
    bucket_name = ''
    bucket = ''
    client = ''

    def __init__(self, s3_bucket=False):

        if s3_bucket != False:
            self.bucket_name = s3_bucket
            self.s3 = True
            self.client = boto3.client('s3')
            resource = boto3.resource('s3')
            self.bucket = resource.Bucket(self.bucket_name)


    def prec_rec_f1(self, y, y_pred, n_classes=4, filename='prec_rec_f1_bars.png'):
        """
        Plot precision recall plot
        """
        logging.debug("Plotting precision recall plot")
        # For each class
        precision = dict()
        recall = dict()
        f1 = dict()

        values = []
        values.append(precision_score(y, y_pred, average='micro'))
        values.append(precision_score(y, y_pred, average='macro'))
        values.append(recall_score(y, y_pred, average='micro'))
        values.append(recall_score(y, y_pred, average='macro'))
        values.append(f1_score(y, y_pred, average='micro'))
        values.append(f1_score(y, y_pred, average='macro'))

        logging.debug('Precision micro avg: {}'.format(values[0]))
        logging.debug('Precision macro avg: {}'.format(values[1]))
        logging.debug('Recall micro avg: {}'.format(values[2]))
        logging.debug('Recall macro avg: {}'.format(values[3]))
        logging.debug('F1 Score micro avg: {}'.format(values[4]))
        logging.debug('F1 Score macro avg: {}'.format(values[5]))

        fig, ax1 = plt.subplots(figsize=(12,8))

        ind = np.arange(6)
        width = 0.35

        rects1 = ax1.bar(ind, values, width, color='#5799c6')

        def autolabel(rects, ax):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.02 * height,
                        '%.2f' % height,
                        ha='center', va='bottom')

        autolabel(rects1, ax1)

        ax1.set_xticks(ind + width/2)
        ax1.set_xticklabels(['Precision micro', 'Precision macro', 'Recall micro', 'Recall macro', 'F1 micro', 'F1 macro'])

        ax1.set_ylabel('Score (AVG over classes)')

        h1, l1 = ax1.get_legend_handles_labels()
        ax1.legend(h1, l1, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                   ncol=3, fancybox=True, shadow=True)
        self._save(plt, filename)

        logging.debug("Saved method comparison chart to " + filename)


    def label_hist(self, y, classes, labels=[], filename='label_hist.png'):
        logging.debug("Plotting label histogram")

        def autolabel(rects, ax):
            """
            Attach a text label above each bar displaying its height
            """
            _max = 0
            for a in rects:
                i = 0
                for rect in a:
                    if i%3 == 0 or True:
                        height = rect.get_height()
                        ax.text(rect.get_x() + rect.get_width()/2., 1.03 * height,
                                '%d' % height,
                                ha='center', va='bottom', rotation=45)
                        if height > _max:
                            _max = height
                    i += 1

            return _max

        fig, ax = plt.subplots(figsize=(8,5))
        bins = np.arange(len(classes) + 1)
        n, bins, rects = plt.hist(y, bins, normed=False, alpha=0.75, align='left', label=labels) #, bins=4)
        _max = autolabel(rects, ax)

        plt.ylim([0, _max*1.2])
        plt.xticks( classes )
        ax.grid(False)
        plt.xlabel('Classes')
        plt.ylabel('Amount')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),  shadow=False, ncol=len(y))
        # plt.legend()
        self._save(plt, filename)

    def hist(self, y, label='', filename='label_hist.png'):
        logging.debug("Plotting histogram")

        fig, ax = plt.subplots(figsize=(8,5))
        plt.hist(y, normed=False, alpha=0.75, align='left', label=label)
        # ax.grid(False)
        plt.xlabel(label)
        plt.ylabel('Amount')
        plt.legend()
        self._save(plt, filename)

    def scree_plot(self, X, filename):
        rows, cols = X.shape
        U, S, V = np.linalg.svd(X)
        eigvals = S**2 / np.cumsum(S)[-1]

        fig = plt.figure(figsize=(8,5))
        sing_vals = np.arange(cols) + 1
        plt.plot(sing_vals, eigvals, 'ro-', linewidth=2)
        plt.title('Scree Plot')
        plt.xlabel('Principal Component')
        plt.ylabel('Eigenvalue')

        leg = plt.legend(['Eigenvalues from SVD'], loc='best', borderpad=0.3,
                         shadow=False, prop=mlp.font_manager.FontProperties(size='small'),
                         markerscale=0.4)
        leg.get_frame().set_alpha(0.4)
        leg.draggable(state=True)

        self._save(plt, filename)

    def model_comp_results(self, results, filename):
        fig, ax1 = plt.subplots(figsize=(12,8))

        x, y1, y2 = [], [], []

        for model,values in results.items():
            x.append(model)
            y1.append(list(values)[0])
            y2.append(list(values)[1])

        ind = np.arange(len(x))
        width = 0.35

        rects1 = ax1.bar(ind, y1, width, color='#5799c6', label='Logistic loss')

        ax2 = ax1.twinx()
        rects2 = ax2.bar(ind + width+0.05, y2, width, color='#ff6450', label='Prediction score')

        def autolabel(rects, ax):
            """
            Attach a text label above each bar displaying its height
            """
            for rect in rects:
                height = rect.get_height()
                ax.text(rect.get_x() + rect.get_width()/2., 1.02 * height,
                        '%.2f' % height,
                        ha='center', va='bottom')

        autolabel(rects1, ax1)
        autolabel(rects2, ax2)

        ax1.set_xticks(ind + width/2)
        ax1.set_xticklabels(x)

        ax1.set_ylabel('Prediction score')
        ax2.set_ylabel('Logistic loss for prediction probabilities')

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax1.legend(h1+h2, l1 + l2, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                   ncol=3, fancybox=True, shadow=True)
        self._save(plt, filename)

        logging.debug("Saved method comparison chart to " + filename)

    def cross_results(self, x, ll, acc, filename):

        fig, ax1 = plt.subplots(figsize=(16,10))

        plt.grid()

        lns1 = ax1.plot(
            x,
            ll,
            c="#27ae61",
            label="Logistic loss")
        ax1.set_ylabel("Logistic Loss")

        ax2 = ax1.twinx()
        lns2 = ax2.plot(
            x,
            acc,
            c="#c1392b",
            label="Accuracy")
        ax2.set_ylabel("Accuracy")

        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper center', frameon=False)
        ax1.set_xlabel("Components")

        logging.debug("Plotted cross validation results to "+ filename)
        self._save(plt, filename)


    def rfc_feature_importance(self, data, filename, feature_names = None,
                               fontsize=12):
        fig, ax = plt.subplots(figsize=(20,8))

        plt.clf()
        plt.rc('font', size=fontsize)

        if feature_names is None:
            feature_names = range(0,len(data))
        else:
            plt.xticks(rotation=90)
            fig.subplots_adjust(bottom=0.3)

        plt.bar(feature_names, data, align='center')
        plt.xlabel('components')
        plt.ylabel('importance')

        self._save(plt, filename)

        logging.debug("Saved feature importance to "+filename)


    def explained_variance(self, pca, filename):

        fig, ax1 = plt.subplots(figsize=(16,10))

        plt.clf()
        plt.xticks(np.arange(0, pca.n_components_, 2))
        plt.grid()

        plt.plot(pca.explained_variance_, linewidth=1)

        #ax2 = ax1.twinx()
        #ax2.plot(pca.explained_variance_ratio_, linewidth=1)

        plt.axis('tight')
        plt.xlabel('Number of components')
        plt.ylabel('Explained variance')
        #ax2.set_ylabel('explained variance ratio')

        self._save(plt, filename)
        logging.debug("Saved explained variance to "+filename)


    def plot_learning(self, metrics, filename,
                      x_label="Iterations", y_label="RMSE"):

        dashes = ['--', '-', '.', '-o-']
        colors = ['brown', 'blue', 'red']
        plt.clf()
        fig = plt.figure(figsize=(26,10))
        #fig, ax1 = plt.subplots(figsize=(26,10))

        # plt.clf()
        # fig.clf()
        # plt.grid()
        plt.suptitle("Loss function across iterations")

        i = 1
        for plot_data in metrics:
            ax = fig.add_subplot(1, len(metrics), i)
            ax.grid()
            j=0
            for metric in plot_data['metrics']:
                dash = dashes[j%len(dashes)]
                color = colors[j%len(colors)]
                ax.plot(np.arange(len(metric['values'])),
                        metric['values'],
                        c=color,
                        linestyle=dash,
                        label=metric['label'])
                j += 1

            if 'x_label' in plot_data:
                ax.set_xlabel(plot_data['x_label'])
            else:
                ax.set_xlabel(x_label)

            if 'y_label' in plot_data:
                ax.set_ylabel(plot_data['y_label'])
            else:
                ax.set_ylabel(x_label)

            #ax.set_yscale('log')

            plt.legend()

            i += 1

        self._save(plt, filename)


    def plot_confusion_matrix(self, y_true, y_pred, classes,
                              normalize=False,
                              cmap=plt.cm.Blues,
                              filename='confusion_matrix.png'):
        """
        This function logging.debugs and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """

        fig, ax = plt.subplots()
        np.set_printoptions(precision=2)
        cm = confusion_matrix(y_true, y_pred)
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            logging.debug("Normalized confusion matrix")
        else:
            logging.debug('Confusion matrix, without normalization')

        logging.debug(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        tick_marks = np.arange(len(classes))
        ax.xaxis.tick_top()
        plt.xticks(tick_marks, classes) #, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        self._save(plt, filename)


    def plot_roc(self, y, y_pred, n_classes=4, filename='roc.png'):
        """
        Plot ROC
        """
        fig, ax1 = plt.subplots(figsize=(12,12))
        plt.clf()

        y = label_binarize(y, classes=np.arange(n_classes))

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(n_classes):
            fpr[i], tpr[i], threshhold = roc_curve(y[:, i], y_pred[:,i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            logging.debug('AUC for class {} is {}'.format(i, roc_auc[i]))
            # logging.debug('Thresholds for class {}: {}'.format(i, threshhold))

        # Compute average ROC curve and ROC area
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

        # Then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])

        # Finally average it and compute AUC
        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.plot([0, 1], [0, 1], 'k--')
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label="Class {0} (AUC: {1:0.2f})".format(i, roc_auc[i]))

        plt.plot(fpr["macro"], tpr["macro"],
                 label='Average (AUC: {0:0.2f})'
                 ''.format(roc_auc["macro"]),
                 color='navy', linestyle=':', linewidth=4)
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc="lower right")
        self._save(plt, filename)

        logging.debug("Saved feature importance to "+filename)


    ################################################################################


    def model_size_comp(self, model_sizes, val_errors, train_errors, filename):
        fig = plt.figure(figsize=(16,10))
        plt.plot(model_sizes, val_errors, linewidth=1.0, label='validation accuracy')
        plt.plot(model_sizes, train_errors, linewidth=1.0, label='training accuray')
        plt.ylabel('accuracy')
        plt.xlabel('model size (r**2, r)')
        plt.axhline(0, color='black', linewidth=0.5)
        plt.legend(frameon=False)
        self._save(plt, filename)


    def plot_activation(self, nactivation1, nactivation2, filename):

        na1, na2 = [], []

        for r in nactivation1.T:
            na1.append(r.mean())

        for r in nactivation2.T:
            na2.append(r.mean())

        fig, ax1 = plt.subplots(figsize=(16,10))

        plt.subplot(211)
        ind = np.arange(len(na1))
        rects1 = plt.bar(ind, na1, color='y')

        plt.subplot(212)
        ind = np.arange(len(na2))
        rects1 = plt.bar(ind, na2, color='r')

        logging.debug("Plotted neuron activations to "+filename)
        self._save(plt, filename)


    ################################################################################

    def plot_nn_perf(self, history,
                     metrics={'MAE': {'mean_absolute_error': 'Mean absolute error'}},
                     filename='nn_perf.png'):

        fig, ax1 = plt.subplots(figsize=(12,8))

        # Get training and test accuracy histories

        #training_accuracy = history.history['acc']
        #test_accuracy = history.history['val_acc']

        #training_loss = history.history['loss']
        #test_loss = history.history['val_loss']

        # Create count of the number of epochs
        dashes = ['--', '-', '.', '-o-']
        lns = []
        axes = {'Loss': ax1}
        axes['Loss'].set_ylabel('Loss')
        epoch_count = range(1, len(history['loss']) + 1)
        lns += axes['Loss'].plot(epoch_count, history['loss'], color='b', linestyle=dashes[0], label='Training loss')
        lns += axes['Loss'].plot(epoch_count, history['val_loss'], color='g', linestyle=dashes[0], label='Validation loss')

        i = 1
        count = 2
        for ax, metrics in metrics.items():
            if ax not in axes:
                axes[ax] = ax1.twinx()
                axes[ax].set_ylabel(ax)

            for metric, lab in metrics.items():
                dash = dashes[i%len(dashes)]
                lns += axes[ax].plot(epoch_count, history[metric], color='b', linestyle=dash, label='Training '+lab)
                lns += axes[ax].plot(epoch_count, history['val_'+metric], color='g', linestyle=dash, label='Validation '+lab)
                count += 2
            i+=1



        #ax2 = ax1.twinx()
        #lns3 = ax2.plot(epoch_count, training_loss, 'y-', label='Training loss')
        #lns4 = ax2.plot(epoch_count, test_loss, 'g-', label='Validation loss')
        #ax2.set_ylabel('Loss')

        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc='upper center', bbox_to_anchor=(0.5, 1.05),
                   ncol=count, frameon=False, shadow=True)

        # ax1.legend(lns, labs, loc=0, frameon=False)
        plt.xlabel('Epoch')

        self._save(plt, filename)

    def plot_model(self, model, filename):
        logging.debug("Plotted model structure to "+filename)
        try:
            if self.s3:
                xfn = basename(filename)
                xffn = '/tmp/'+xfn
                plot_model(model, to_file=xffn, show_shapes=True)
                self.bucket.upload_file(xffn, filename)
            else:
                plot_model(model, to_file=filename, show_shapes=True)
        except:
            logging.debug("...FAILED")


    def plot_feature_map(self, model, layer_id, activation_id, X, filename, n=256, ax=None, **kwargs):
        """
        """
        import keras.backend as K
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # fig, ax = plt.subplots(figsize=(16,16))

        layer = model.layers[layer_id]
        activation_layer = model.layers[activation_id]
        logging.debug(layer.name)

        # Get data
        get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
        get_activations = K.function([model.layers[0].input, K.learning_phase()], [activation_layer.output,])
        output = get_output([X, 0])[0]
        activations = get_activations([X, 0])[0]

        # Set default matplotlib parameters
        if not 'interpolation' in kwargs.keys():
            kwargs['interpolation'] = "none"

        if not 'cmap' in kwargs.keys():
            kwargs['cmap'] = "gray"

        fig = plt.figure(figsize=(20, 20))

        # Compute nrows and ncols for images
        n_mosaic = activations.shape[1]
        nrows = int(np.round(np.sqrt(n_mosaic)))
        ncols = int(nrows)
        if (nrows ** 2) < n_mosaic:
            ncols +=1

        # Compute nrows and ncols for mosaics
        if activations[0].shape[0] < n:
            n = activations[0].shape[0]

        nrows_inside_mosaic = int(np.round(np.sqrt(n)))
        ncols_inside_mosaic = int(nrows_inside_mosaic)

        if nrows_inside_mosaic ** 2 < n:
            ncols_inside_mosaic += 1


        for i, feature_map in enumerate(np.transpose(output)):
            ax = fig.add_subplot(ncols, nrows, i+1)
            mosaic = ax.hist(feature_map)

            ax.set_title("Historgram #{} \nof layer#{} \ncalled '{}' \nof type {} ".format(i, layer_id,layer.name, layer.__class__.__name__))

        fig.tight_layout()
        self._save(fig, filename)
        return fig

    def plot_all_feature_maps(self, model, X, layers, activations, path='.', n=256, ax=None, **kwargs):
        """
        """

        n = len(layers)
        i = 0
        for layer, activation in zip(layers, activations):
            try:
                logging.debug("Plotting feature map for layer {} with activation {}".format(layer, activation))
                filename=path+'/feature_map_'+str(i)+'.png'
                fig = self.plot_feature_map(model, layer, activation, X, filename=filename, n=n, ax=ax, **kwargs)
                i += 1
            except:
                logging.debug("...FAILED")

        self._save(fig, filename)


    def _save(self, p, filename):
        if self.s3:
            xfn = basename(filename)
            xffn = '/tmp/'+xfn
            p.savefig(xffn)
            self.bucket.upload_file(xffn, filename)
        else:
            logging.info('Saved file {}'.format(filename))
            p.savefig(filename)

        plt.close()


    # TRAINS ####################################################

    def hist_all_delays(self, df, filename):
        """
        """

        axs = df.hist(alpha=0.5, range=(0,10), bins=11, density=True, align='left')

        for row in axs:
            for ax in row:
                ax.set_xticks(np.arange(11))

        fig = axs[0][0].get_figure()
        self._save(fig, filename)

    def plot_delays(self, df, filename):
        """
        """
        plt.clf()
        axs = df.plot(alpha=0.5, subplots=True, figsize=(60,80))

#        import matplotlib.dates as mdates

#        years = mdates.YearLocator()   # every year
#        months = mdates.MonthLocator()  # every month
        # days = mdates.DayLocator()

#        axs[0].xaxis.set_major_locator(years)
#        axs[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

#        axs[0].xaxis.set_minor_locator(ticker.FixedLocator(minors))
#        axs[0].xaxis.set_minor_formatter(mdates.DateFormatter('%H'))
        #axs[0].xaxis.set_minor_locator(days)
#        print(axs[0].get_xticklabels())

        #axs[0].xaxis.set_major_locator(MaxNLocator(nbins=20, prune='upper'))

        fig = axs[0].get_figure()
        fig.autofmt_xdate()
        for ax in axs:
            ax.grid(True, which='major', axis='both')
            ax.grid(True, which='minor', axis='x')
            ax.tick_params(axis = 'both', which = 'major', labelsize = 50)
            ax.tick_params(axis = 'both', which = 'minor', labelsize = 50)

        params = {'legend.fontsize': 50}
        plt.rcParams.update(params)

        plt.ylabel('Delay [minutes]', fontsize=50)

        self._save(fig, filename)

    def heatmap_train_station_delays(self, df, locs, filename, label='Day of year'):
        """
        """
        plt.clf()

        ratio = df.shape[0]/df.shape[1]
        width = 120
        height = int(width*ratio)
        if height < 10: height = 10

        fontsize = 50
        if height < 60: fontsize = 25

        fig, ax = plt.subplots(figsize=(width,height))
        cax = ax.matshow(df, interpolation=None, aspect='auto')

        labels = []
        for name in df.columns.get_level_values('location id'):
            labels.append(locs[name]['name'])

        loc = ticker.MultipleLocator(base=1.0)
        ax.xaxis.set_major_locator(loc)
        ax.set_xticklabels(['']+labels)

        ax.tick_params(axis = 'x', which = 'major', labelsize = 16)
        ax.tick_params(axis = 'y', which = 'major', labelsize = fontsize)
        ax.tick_params(axis='x', rotation=90)

        cbar = fig.colorbar(cax)
        for font_objects in cbar.ax.yaxis.get_ticklabels():
            font_objects.set_size(fontsize)

        plt.ylabel(label, fontsize=fontsize)
        plt.xlabel('Station', fontsize=fontsize)

        plt.subplots_adjust(left=0.1)

        logging.info('Saving file {}...'.format(filename))
        plt.savefig(filename)
        plt.close()


    def _split_to_parts(self, times, data, gap=None):
        """
        Split time list to gaps
        """
        if gap is None:
            gaps = []
            prev = times[0]
            for t in times[1:]:
                dt = t - prev
                gaps.append(dt.total_seconds())
                prev = t
            gap = max(set(gaps), key=gaps.count)

        splits = []
        prev = times[0]
        temp_data = []
        for j in np.arange(0,len(data)+1):
            temp_data.append([])

        i = 0
        for t in times:
            dt = t - prev
            if dt.total_seconds() > gap:
                splits.append(copy.deepcopy(temp_data))
                for j in np.arange(0,len(data)):
                    temp_data[j+1] = []
                temp_data[0] = []

            temp_data[0].append(t)
            for j in np.arange(0,len(data)):
                if data[j] is not None:
                    temp_data[j+1].append(data[j][i])
                else:
                    temp_data[j+1].append(None)

            prev = t
            i += 1
        splits.append(copy.deepcopy(temp_data))
        return splits

    def plot_delay(self, all_times, all_delay=None, all_pred_delay=None,
                   heading='Delay', filename='delay.png',
                   all_low=None, all_high=None):
        """
        Plot delay and predicted delay over time
        """
        plt.clf()

        plt.rcParams.update({'font.size': 16})

        splits = self._split_to_parts(all_times, [all_delay, all_pred_delay, all_low, all_high], 2592000)
        logging.info('Data have {} splits'.format(len(splits)))
        fig, axes = plt.subplots(1, len(splits), figsize=(28,10))

        years = mdates.YearLocator()   # every year
        months = mdates.MonthLocator()  # every month
        days = mdates.DayLocator()
        hours = mdates.HourLocator()
        yearsFmt = mdates.DateFormatter('%m')

        plt.grid()
        max_ = 0

        for i in range(0, len(splits)):
            if len(splits) > 1:
                ax = axes[i]
            else:
                ax = axes

            times, delay, pred_delay, low, high = splits[i]
            max_ = 0
            if low[0] is not None and high[0] is not None:
                ax.fill_between(times,
                                high,
                                low,
                                facecolor="#ff7a7a",
                                alpha=0.5,
                                joinstyle='round',
                                capstyle='round',
                                label='Predicted delay 10-90 % quantile')

                if max(high) > max_:
                    max_ = max(high)

            if pred_delay is not None:
                ax.plot(times,
                         pred_delay,
                         c="#ff7a7a",
                         linewidth=0.5,
                         label="Predicted delay")
                if max(pred_delay) > max_:
                    max_ = max(pred_delay)

            if delay is not None:
                ax.plot(times,
                         delay,
                         c="#126cc5",
                         linewidth=0.5,
                         label="True delay")
                if max(delay) > max_:
                    max_ = max(delay)

            ax.format_xdata = mdates.DateFormatter('%Y %m %d')
            ax.set_xlabel("Time")
            ax.set_ylabel("Delay [minutes]")
            ax.title.set_text(heading)

            ax.grid(True)
            fig.autofmt_xdate()
            ax.legend()

        for i in range(0, len(splits)):
            if len(splits) > 1:
                ax = axes[i]
            else:
                ax = axes
            if max_ <= 100:
                ax.set_ylim([0,100])
            else:
                ax.set_ylim([0,max_])

        self._save(plt, filename)

    def plot_learning_over_time(self, times, rmse=None, mae=None,
                                r2=None, heading='Model error development over time',
                                filename='error.png'):

        fig, ax1 = plt.subplots(figsize=(16,10))

        years = mdates.YearLocator()
        months = mdates.MonthLocator()
        days = mdates.DayLocator()
        hours = mdates.HourLocator()
        yearsFmt = mdates.DateFormatter('%d/%m %Y')

        lns = []
        if rmse is not None:
            lns += plt.plot(times,
                            rmse,
                            c="#27ae61",
                            label="RMSE")

        if mae is not None:
            lns += plt.plot(times,
                            mae,
                            c="#c1392b",
                            label="MAE")

        plt.xlabel('Date of delay data')
        plt.ylabel('Error')

        if r2 is not None:
            plt.twinx()
            plt.ylabel(r'$R^2$ Score')
            lns += plt.plot(times,
                            r2,
                            c="#223999",
                            label=r'$R^2$ Score')

        plt.title(heading)

        ax1.xaxis.set_major_locator(months)
        ax1.xaxis.set_major_formatter(yearsFmt)
        #ax1.xaxis.set_minor_locator(days)

        plt.format_xdata = mdates.DateFormatter('%Y %m %d')
        plt.grid(True)

        fig.autofmt_xdate()

        labs = [l.get_label() for l in lns]
        plt.legend(lns, labs, frameon=True)
        #plt.legend()
        self._save(plt, filename)


    def plot_svga(self, m, filename):
        """
        Plot Sparse Variational Gaussian approximation performance.

        https://gpflow.readthedocs.io/en/latest/notebooks/multiclass.html#Sparse-Variational-Gaussian-approximation
        """

        f = plt.figure(figsize=(12,6))
        a1 = f.add_axes([0.05, 0.05, 0.9, 0.6])
        a2 = f.add_axes([0.05, 0.7, 0.9, 0.1])
        a3 = f.add_axes([0.05, 0.85, 0.9, 0.1])

        xx = np.linspace(m.X.read_value().min(), m.X.read_value().max(), 200).reshape(-1,1)
        mu, var = m.predict_f(xx)
        mu, var = mu.copy(), var.copy()
        p, _ = m.predict_y(xx)

        a3.set_xticks([])
        a3.set_yticks([])

        a3.set_xticks([])
        a3.set_yticks([])

        i=0
        x = m.X.read_value()[m.Y.read_value().flatten()==i]
        points, = a3.plot(x, x*0, '.')
        color=points.get_color()
        a1.plot(xx, mu[:,i], color=color, lw=2)
        a1.plot(xx, mu[:,i] + 2*np.sqrt(var[:,i]), '--', color=color)
        a1.plot(xx, mu[:,i] - 2*np.sqrt(var[:,i]), '--', color=color)
        a2.plot(xx, p[:,i], '-', color=color, lw=2)

        a2.set_ylim(-0.1, 1.1)
        a2.set_yticks([0, 1])
        a2.set_xticks([])

        self._save(plt, filename)
