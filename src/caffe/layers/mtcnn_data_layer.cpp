#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/mtcnn_data_layer.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

template <typename Dtype>
MTCNNDataLayer<Dtype>::MTCNNDataLayer(const LayerParameter& param)
  : BasePrefetchingDataLayer<Dtype>(param),
    reader_(param) {
}

template <typename Dtype>
MTCNNDataLayer<Dtype>::~MTCNNDataLayer() {
  this->StopInternalThread();
}

template <typename Dtype>
void MTCNNDataLayer<Dtype>::DataLayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    const int batch_size = this->layer_param_.data_param().batch_size();
    // Read a data point, and use it to initialize the top blob.
    MTCNNDatum& datum = *(reader_.full().peek());

    //data, label, roi, pts
    // Use data_transformer to infer the expected blob shape from datum.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum.datum());
    this->transformed_data_.Reshape(top_shape);
    // Reshape top[0] and prefetch_data according to the batch_size.
    top_shape[0] = batch_size;

    //data
    top[0]->Reshape(top_shape);
    for (int i = 0; i < this->prefetch_.size(); ++i) {
        this->prefetch_[i]->data_.Reshape(top_shape);
    }
    LOG(INFO) << "output data size: " << top[0]->num() << ","
        << top[0]->channels() << "," << top[0]->height() << ","
        << top[0]->width();

    // label
    if(this->output_labels_){
        vector<int> label_shape(2);
        label_shape[0] = batch_size;
        label_shape[1] = 1;
        top[1]->Reshape(label_shape);
        for (int i = 0; i < this->prefetch_.size(); ++i) {
            this->prefetch_[i]->label_.Reshape(label_shape);
        }
    }

    //roi
    if(this->output_roi_){
        vector<int> roi_shape(2);
        roi_shape[0] = batch_size;
        roi_shape[1] = 4;
        top[2]->Reshape(roi_shape);
        for (int i = 0; i < this->prefetch_.size(); ++i) {
            this->prefetch_[i]->roi_.Reshape(roi_shape);
        }
    }

    //pts
    if (this->output_pts_){
        vector<int> pts_shape(2);
        pts_shape[0] = batch_size;
        pts_shape[1] = datum.pts_size();
        top[3]->Reshape(pts_shape);
        for (int i = 0; i < this->prefetch_.size(); ++i) {
            this->prefetch_[i]->pts_.Reshape(pts_shape);
        }
    }

}

// This function is called on prefetch thread
template<typename Dtype>
void MTCNNDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());

    // Reshape according to the first datum of each batch
    // on single input batches allows for inputs of varying dimension.
    const int batch_size = this->layer_param_.data_param().batch_size();
    MTCNNDatum& datum = *(reader_.full().peek());
    // Use data_transformer to infer the expected blob shape from datum.
    vector<int> top_shape = this->data_transformer_->InferBlobShape(datum.datum());
    this->transformed_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size.
    top_shape[0] = batch_size;
    batch->data_.Reshape(top_shape);

    Dtype* top_data = batch->data_.mutable_cpu_data();

    for (int item_id = 0; item_id < batch_size; ++item_id) {
        timer.Start();
        // get a datum
        MTCNNDatum& datum = *(reader_.full().pop("Waiting for data"));
        const Datum& data = datum.datum();
        //printf("****************datum.pts_size() = %d\n", datum.pts_size());

        read_time += timer.MicroSeconds();
        timer.Start();
        // Apply data transformations (mirror, scale, crop...)
        int offset = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(top_data + offset);
        this->data_transformer_->Transform(datum.datum(), &(this->transformed_data_));
        // Copy label.
        if (this->output_labels_){
            Dtype* top_label = batch->label_.mutable_cpu_data();
            top_label[item_id] = data.label();
        }

        //copy roi
        if(this->output_roi_){
            Dtype* top_roi = batch->roi_.mutable_cpu_data();
            top_roi[item_id * 4 + 0] = datum.roi().xmin();
            top_roi[item_id * 4 + 1] = datum.roi().ymin();
            top_roi[item_id * 4 + 2] = datum.roi().xmax();
            top_roi[item_id * 4 + 3] = datum.roi().ymax();
        }


        //printf("datum.pts_size() = %d\n", datum.pts_size());
        //copy pts
        if (this->output_pts_){
            Dtype* top_pts = batch->pts_.mutable_cpu_data();
            for (int ptsi = 0; ptsi < datum.pts_size(); ++ptsi)
                top_pts[item_id * datum.pts_size() + ptsi] = datum.pts(ptsi);
        }

        trans_time += timer.MicroSeconds();
        reader_.free().push(const_cast<MTCNNDatum*>(&datum));
    }
    timer.Stop();
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(MTCNNDataLayer);
REGISTER_LAYER_CLASS(MTCNNData);

}  // namespace caffe
