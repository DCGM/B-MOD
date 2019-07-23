# B-MOD
Models from paper M. Kišš, M. Hradiš, and O. Kodym, “Brno Mobile OCR Dataset” in *2019 15th IAPR International Conference on Document Analysis and Recognition (ICDAR)*, IEEE, 2019.
([arxiv](https://arxiv.org/abs/1907.01307))

We also provide a python script for transcribing text lines with our proposed networks. 
The script runs with parameters specifying directory with image to transcribe and also used network. 
For example

```python3 transcribe.py --input-dir=/tmp/my_dir/ --net=LSTM```

transcribes all `.jpg` and `.png` files in `/tmp/my_dir` using neural network with recurrent layers (network stored in `lstm_net/`).
If you want to transcribe lines using netowrk without recurrent layers, use `--net=CONV` parameter instead.
