# 概要
JPEGなどの8x8ブロックごとに離散コサイン変換された画像の高速な変換を行います.
行うことのできる変換は以下の4つ.
- 平行移動
- 90°単位の回転
- 鏡像変換(反転)
- 一部のマスク

# インストール方法
```shell
$pip install git+https://github.com/kanpurin/dctimagetransform
```

# 使い方
```python
import dct_image_transform.dct as dct
import dct_image_transform.translation as translation

image = "画像を表す2次元配列"

dct_image = dct.dct2(image) # 画像を8x8ブロックごとに離散コサイン変換
tf_image = translation.translation(dct_image,100) # DCT画像を平行移動
```

# 要件
[numpy](https://numpy.org/)が必要です. 以下のコマンドでインストールしてください.
```shell
$pip install numpy
```

# 速度
サイズが1024x1024である画像に各変換を100回行ったときの平均実行時間は以下のようになっています.
- 通常変換：離散コサイン変換された画像を逆変換して元に戻し通常の変換を行ってまた離散コサイン変換するまでにかかる時間.
- 直接変換：離散コサイン変換された画像のままでの変換(本実装はこちら)にかかる時間.
- DCT：通常変換での離散コサイン変換および逆変換にかかる時間.

<img src="https://user-images.githubusercontent.com/56508025/104653133-8a2f4380-56fd-11eb-83fd-df127cd6c45e.jpg"/>

# 理論
ある8x8のブロック<img src="https://latex.codecogs.com/gif.latex?G"/>を離散コサイン変換すると, 変換行列を<img src="https://latex.codecogs.com/gif.latex?D"/>として<img src="https://latex.codecogs.com/gif.latex?DGD^T"/>と表される.<br>

工事中