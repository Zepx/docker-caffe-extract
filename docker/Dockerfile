FROM bvlc/caffe:gpu

WORKDIR /caffe
COPY python/extract_weight.py /usr/local/bin/extract_weight.py

VOLUME ["/caffe"]
CMD ['python', 'extract_weight.py']
