## これは何

入力動画に

- 顔検出結果

- 目の位置

- 人の視線

を描画した結果を返すアプリケーション

## 使用方法

1. dockerイメージをビルド

```
docker-compose build
```

2. 入力したい動画をinputディレクトリに入れて

```
docker-compose run --rm gaze_estimation python3 main.py --input_file_name <入力画像名(デフォルトはdemo1.mp4)> --output_file_name <出力結果動画名(デフォルトはoutput.mp4)>
```

## アルゴリズム

以下のようなアルゴリズムで視線ベクトルを取得して描画している。

1. 顔を検出(https://docs.openvino.ai/2019_R1/_face_detection_retail_0004_description_face_detection_retail_0004.html)

2. 以下の情報を取得

    - 顔ランドマーク(https://docs.openvino.ai/2019_R1/_facial_landmarks_35_adas_0002_description_facial_landmarks_35_adas_0002.html): 目の位置を取得するために使用

    - 頭の角度(https://docs.openvino.ai/2019_R1/_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)

3. (顔画像から切り抜いた左目, 顔画像から切り抜いた右目, 頭部の角度)を入力として、視線ベクトル取得(https://docs.openvino.ai/2019_R1/_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

## 改善案

視線ベクトルを数フレームの平均などにすればもう少し正確になる可能性がある。

## 参考

- https://qiita.com/oozzZZZZ/items/1e68a7572bc5736d474e, https://sgrsn1711.hatenablog.com/entry/2018/07/01/182207 理論背景的にはこれらの記事が近い?

- https://github.com/yas-sim/gaze-estimation-with-laser-sparking 参考ソースコード

- https://github.com/kyotovision-public/dynamic-3d-gaze-from-afar 試していないが、京大発表の「体全体」と「頭」から視線ベクトルを取得する方法もあるよう(店などで使用する場合は、こちらがよさそう)
