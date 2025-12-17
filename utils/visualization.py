import cv2
import numpy as np

def draw_bounding_boxes(image, boxes, labels, scores, label_map, color=(0, 0, 255)):
    result_image = image.copy()
    overlay = image.copy()
    
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        if scores is not None:
            text = f'{label_map[labels[idx]]}: {scores[idx]:.2f}'
        else:
            text = label_map[labels[idx]]
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_PLAIN, 1, 1)
        text_w, text_h = text_size
        cv2.rectangle(overlay, (x1, y1), (x1 + 10 + text_w, y1 + 10 + text_h), 
                     (255, 255, 255), -1)
        cv2.putText(result_image, text, (x1 + 5, y1 + 15),
                   cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
        cv2.putText(overlay, text, (x1 + 5, y1 + 15),
                   cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)
    
    cv2.addWeighted(overlay, 0.7, result_image, 0.3, 0, result_image)
    return result_image