# LSUN Church Dataset
git clone https://github.com/fyu/lsun.git
python3 lsun/download.py -c church_outdoor
rm -rf lsun
unzip church_outdoor_train_lmdb.zip
