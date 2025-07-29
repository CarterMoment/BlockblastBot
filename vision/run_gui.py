import cv2
from capture import get_blockblast_frame_from_quicktime
from detect_board import extract_board_matrix

def draw_matrix_on_image(image, matrix):
    h, w = image.shape[:2]
    cell_h = h // 8
    cell_w = w // 8

    for i in range(8):
        for j in range(8):
            x = j * cell_w + 5
            y = i * cell_h + 30
            val = str(matrix[i][j])
            cv2.putText(image, val, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)
    return image

def main():
    while True:
        frame = get_blockblast_frame_from_quicktime()
        if frame is None:
            print("[WARN] Block Blast window not detected.")
            continue

        matrix, board_img = extract_board_matrix(frame)
        annotated = draw_matrix_on_image(board_img.copy(), matrix)

        cv2.imshow("Block Blast Matrix", annotated)
        key = cv2.waitKey(200)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()