import cv2
import numpy as np

def create_point_cloud(depth_image, color_image, intrinsic_matrix):
    rows, cols = depth_image.shape
    points = []

    for v in range(rows):
        for u in range(cols):
            Z = depth_image[v, u] / 1000.0  # Convert depth from millimeters to meters
            X = (u - intrinsic_matrix[0, 2]) * Z / intrinsic_matrix[0, 0]
            Y = (v - intrinsic_matrix[1, 2]) * Z / intrinsic_matrix[1, 1]

            color = color_image[v, u]
            points.append([X, Y, Z, color[2], color[1], color[0]])

    point_cloud = np.array(points, dtype=np.float32)
    return point_cloud

def write_pcd_file(point_cloud, file_path):
    with open(file_path, 'w') as f:
        f.write("# .PCD v0.7 - Point Cloud Data\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z rgb\n")
        f.write("SIZE 4 4 4 4\n")
        f.write("TYPE F F F F\n")
        f.write("COUNT 1 1 1 1\n")
        f.write("WIDTH {}\n".format(len(point_cloud)))
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write("POINTS {}\n".format(len(point_cloud)))
        f.write("DATA ascii\n")
        for point in point_cloud:
            f.write("{} {} {} {:d}\n".format(point[0], point[1], point[2], int((point[3] << 16) + (point[4] << 8) + point[5])))

def main():
    # Open a video capture object (0 for default camera, you can change it based on your setup)
    cap = cv2.VideoCapture(0)

    # Load your camera intrinsic matrix
    intrinsic_matrix = np.array([[fx, 0, cx],
                                 [0, fy, cy],
                                 [0, 0, 1]])

    while True:
        # Read a frame from the video stream
        ret, frame = cap.read()

        # Convert the frame to grayscale (you can modify this based on your actual video)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Create 3D point cloud
        point_cloud = create_point_cloud(gray, frame, intrinsic_matrix)

        # Save point cloud to .pcd file
        write_pcd_file(point_cloud, "output_cloud.pcd")

        # Display the original frame
        cv2.imshow('Original Frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
