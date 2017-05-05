FROM bvlc/caffe:gpu

WORKDIR /caffe
COPY python/extract_weights.py /usr/local/bin/extract_weights.py

VOLUME ["/caffe"]
CMD ['python', '/usr/local/bin/extract_weight.py']
