from ultralytics import YOLO

TRAIN = False
RUN_VAL = False
RUN_PREDICT = True
RUN_EXPORT = False

if TRAIN:
    model = YOLO("yolo26n.pt")
    model.train(data="coco8.yaml", epochs=100, imgsz=640, device="cpu")
    model = YOLO("/Users/emmanuel/Documents/wear_wise/runs/detect/train3/weights/best.pt")
else:
    model = YOLO("/Users/emmanuel/Documents/wear_wise/runs/detect/train3/weights/best.pt")

if RUN_VAL:
    metrics = model.val()

if RUN_PREDICT:
    results = model(
        "/Users/emmanuel/Downloads/IMG_9679.HEIC",
        conf=0.1
    )
    r = results[0]

    print("----Checking What is in results list------")
    print(r.boxes)
    print("-------Check Finished----------")

    if len(r.boxes) == 0:
        print("No detections found.")
    else:
        pairs = sorted(
            zip(r.boxes.cls.tolist(), r.boxes.conf.tolist()),
            key=lambda x: x[1],
            reverse=True
        )[:3]
        print("Top 3 predictions:")
        for i, (cls_id, conf) in enumerate(pairs, start=1):
            print(f"{i}. {r.names[int(cls_id)]} ({conf:.3f})")
        r.show()

if RUN_EXPORT:
    path = model.export(format="onnx")