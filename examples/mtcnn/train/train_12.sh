cd /home/vincent/github/caffe
./build/tools/caffe train \
--solver="/home/vincent/github/caffe/examples/mtcnn/train/solver-12.prototxt" \
--gpu 0 2>&1 | tee /home/vincent/github/caffe/jobs/mtcnn/pnet.log


# train --solver="/home/vincent/github/caffe/examples/mtcnn/train/solver-12.prototxt"
--snapshot="/home/vincent/github/caffe/models/mtcnn/mtcnn_pnet_iter_500000.solverstate"
--gpu 0
