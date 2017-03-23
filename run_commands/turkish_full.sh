#!/bin/bash
THEANO_FLAGS="device=gpu,floatX=float32" python ../semi_models/run.py -bidirectional -worddrop 0.4 -lang turkish -kl_thres 0.2 -start_val 15000 -epochs 120 -add_uns 0.0