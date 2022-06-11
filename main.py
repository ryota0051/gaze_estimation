import math
import argparse
from pathlib import Path

import numpy as np
import cv2
from openvino.inference_engine import IECore

# https://docs.openvino.ai/2019_R1/_face_detection_retail_0004_description_face_detection_retail_0004.html
model_det = "face-detection-retail-0004"
# https://docs.openvino.ai/2019_R1/_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html
model_hp = "head-pose-estimation-adas-0001"
# https://docs.openvino.ai/2019_R1/_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html
model_gaze = "gaze-estimation-adas-0002"
# https://docs.openvino.ai/2019_R1/_facial_landmarks_35_adas_0002_description_facial_landmarks_35_adas_0002.html
model_lm = "facial-landmarks-35-adas-0002"

model_det = "./models/intel/" + model_det + "/FP16/" + model_det
model_hp = "./models/intel/" + model_hp + "/FP16/" + model_hp
model_gaze = "./models/intel/" + model_gaze + "/FP16/" + model_gaze
model_lm = "./models/intel/" + model_lm + "/FP16/" + model_lm

_H = 2
_W = 3
_X = 0
_Y = 1


def draw_gaze_line(img, coord1, coord2):
    cv2.arrowedLine(img, coord1, coord2, (0, 0, 255), 2)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_name", default="demo1.mp4")
    parser.add_argument("--output_file_name", default="output.mp4")
    return parser.parse_args()


def main():
    # 引数取得
    args = get_args()
    # 入力用動画読み込み
    src = str(Path("input") / args.input_file_name)
    capture = cv2.VideoCapture(src)
    num_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not capture.isOpened():
        print("Video could not open")
        exit(-1)

    boundary_box_flag = True

    # 動画準備
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    size = (width, height)
    fmt = cv2.VideoWriter_fourcc("m", "p", "4", "v")
    dst = str(Path("output") / args.output_file_name)
    writer = cv2.VideoWriter(dst, fmt, fps, size)

    # 顔検出器取得
    ie = IECore()

    net_det = ie.read_network(model=model_det + ".xml", weights=model_det + ".bin")
    input_name_det = next(iter(net_det.input_info))
    input_shape_det = net_det.input_info[input_name_det].tensor_desc.dims
    out_name_det = next(iter(net_det.outputs))
    exec_net_det = ie.load_network(network=net_det, device_name="CPU", num_requests=1)
    del net_det

    # ランドマーク検出器取得
    net_lm = ie.read_network(model=model_lm + ".xml", weights=model_lm + ".bin")
    input_name_lm = next(iter(net_lm.input_info))
    input_shape_lm = net_lm.input_info[input_name_lm].tensor_desc.dims
    out_name_lm = next(iter(net_lm.outputs))
    exec_net_lm = ie.load_network(network=net_lm, device_name="CPU", num_requests=1)
    del net_lm

    # 頭部傾き検出器取得
    net_hp = ie.read_network(model=model_hp + ".xml", weights=model_hp + ".bin")
    input_name_hp = next(iter(net_hp.input_info))
    exec_net_hp = ie.load_network(network=net_hp, device_name="CPU", num_requests=1)
    del net_hp

    # 視線検出器取得
    net_gaze = ie.read_network(model=model_gaze + ".xml", weights=model_gaze + ".bin")
    input_shape_gaze = [1, 3, 60, 60]
    exec_net_gaze = ie.load_network(network=net_gaze, device_name="CPU")
    del net_gaze

    frame_count = 0
    print(f"num frame: {num_frames}")
    while True:
        ret, img = capture.read()
        if not ret:
            break

        out_img = img.copy()

        img1 = cv2.resize(img, (input_shape_det[_W], input_shape_det[_H]))
        img1 = img1.transpose((2, 0, 1))
        img1 = img1.reshape(input_shape_det)
        res_det = exec_net_det.infer(inputs={input_name_det: img1})

        for obj in res_det[out_name_det][0][0]:
            if obj[2] > 0.75:
                xmin = abs(int(obj[3] * img.shape[1]))
                ymin = abs(int(obj[4] * img.shape[0]))
                xmax = abs(int(obj[5] * img.shape[1]))
                ymax = abs(int(obj[6] * img.shape[0]))
                face = img[ymin:ymax, xmin:xmax]
                if boundary_box_flag:
                    cv2.rectangle(out_img, (xmin, ymin), (xmax, ymax), (255, 255, 0), 2)

                # 顔ランドマーク検出
                face1 = cv2.resize(face, (input_shape_lm[_W], input_shape_lm[_H]))
                face1 = face1.transpose((2, 0, 1))
                face1 = face1.reshape(input_shape_lm)
                res_lm = exec_net_lm.infer(inputs={input_name_lm: face1})
                lm = res_lm[out_name_lm][0][:8].reshape(4, 2)

                # 頭の傾き検出 (yaw=Y, pitch=X, role=Z)
                # https://journals.plos.org/plosone/article/figure?id=10.1371/journal.pone.0192767.g001
                # みたいなイメージ
                res_hp = exec_net_hp.infer(inputs={input_name_hp: face1})
                yaw = res_hp["angle_y_fc"][0][0]
                pitch = res_hp["angle_p_fc"][0][0]
                roll = res_hp["angle_r_fc"][0][0]

                # 左右の目のサイズと中心位置取得
                # 座標は、https://docs.openvino.ai/2019_R1/_facial_landmarks_35_adas_0002_description_facial_landmarks_35_adas_0002.html を参照
                eye_sizes = [
                    abs(int((lm[0][_X] - lm[1][_X]) * face.shape[1])),
                    abs(int((lm[3][_X] - lm[2][_X]) * face.shape[1])),
                ]
                eye_centers = [
                    [
                        int(((lm[0][_X] + lm[1][_X]) / 2 * face.shape[1])),
                        int(((lm[0][_Y] + lm[1][_Y]) / 2 * face.shape[0])),
                    ],
                    [
                        int(((lm[3][_X] + lm[2][_X]) / 2 * face.shape[1])),
                        int(((lm[3][_Y] + lm[2][_Y]) / 2 * face.shape[0])),
                    ],
                ]
                if eye_sizes[0] < 4 or eye_sizes[1] < 4:
                    continue

                ratio = 0.7
                eyes = []
                for i in range(2):
                    # Crop eye images
                    x1 = int(eye_centers[i][_X] - eye_sizes[i] * ratio)
                    x2 = int(eye_centers[i][_X] + eye_sizes[i] * ratio)
                    y1 = int(eye_centers[i][_Y] - eye_sizes[i] * ratio)
                    y2 = int(eye_centers[i][_Y] + eye_sizes[i] * ratio)
                    eyes.append(
                        cv2.resize(
                            face[y1:y2, x1:x2].copy(),
                            (input_shape_gaze[_W], input_shape_gaze[_H]),
                        )
                    )

                    # Draw eye boundary boxes
                    if boundary_box_flag:
                        cv2.rectangle(
                            out_img,
                            (x1 + xmin, y1 + ymin),
                            (x2 + xmin, y2 + ymin),
                            (0, 255, 0),
                            2,
                        )

                    # rotate eyes around Z axis to keep them level
                    if roll != 0.0:
                        rotMat = cv2.getRotationMatrix2D(
                            (
                                int(input_shape_gaze[_W] / 2),
                                int(input_shape_gaze[_H] / 2),
                            ),
                            roll,
                            1.0,
                        )
                        eyes[i] = cv2.warpAffine(
                            eyes[i],
                            rotMat,
                            (input_shape_gaze[_W], input_shape_gaze[_H]),
                            flags=cv2.INTER_LINEAR,
                        )
                    eyes[i] = eyes[i].transpose((2, 0, 1))
                    eyes[i] = eyes[i].reshape((1, 3, 60, 60))

                hp_angle = [yaw, pitch, 0]
                res_gaze = exec_net_gaze.infer(
                    inputs={
                        "left_eye_image": eyes[0],
                        "right_eye_image": eyes[1],
                        "head_pose_angles": hp_angle,
                    }
                )
                gaze_vec = res_gaze["gaze_vector"][0]
                gaze_vec_norm = gaze_vec / np.linalg.norm(gaze_vec)

                vcos = math.cos(math.radians(roll))
                vsin = math.sin(math.radians(roll))
                tmpx = gaze_vec_norm[0] * vcos + gaze_vec_norm[1] * vsin
                tmpy = -gaze_vec_norm[0] * vsin + gaze_vec_norm[1] * vcos
                gaze_vec_norm = [tmpx, tmpy]

                # Store gaze line coordinations
                for i in range(2):
                    start_x = eye_centers[i][_X] + xmin
                    start_y = eye_centers[i][_Y] + ymin
                    coord1 = (start_x, start_y)
                    end_x = (
                        eye_centers[i][_X] + xmin + int((gaze_vec_norm[0] + 0.0) * 100)
                    )
                    end_y = (
                        eye_centers[i][_Y] + ymin - int((gaze_vec_norm[1] + 0.0) * 100)
                    )
                    coord2 = (end_x, end_y)
                    draw_gaze_line(out_img, coord1, coord2)

        writer.write(out_img)
        frame_count += 1
        if frame_count % 10 == 0:
            print(f"{frame_count} / {num_frames} end")
    writer.release()
    capture.release()


if __name__ == "__main__":
    main()
