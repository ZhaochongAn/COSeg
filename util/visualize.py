import numpy as np
import open3d as o3d
import os
import argparse


def crop(pcd):
    vis = o3d.visualization.VisualizerWithEditing()

    print("Demo for manual geometry cropping")
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry")
    print("5) Press 'S' to save the selected geometry")
    print("6) Press 'F' to switch to freeview mode")
    vis.create_window(window_name="crop", width=1440, height=1080)
    vis.add_geometry(pcd)
    # opt.show_coordinate_frame = True
    vis.run()  # user picks points

    view_point = vis.get_view_control().convert_to_pinhole_camera_parameters()
    vis.destroy_window()
    cropped_geometry = vis.get_cropped_geometry()
    return cropped_geometry, view_point


def support_vis(file_path, sampled_class=None):
    print(file_path)
    data = np.load(file_path)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6])
    label = np.load(file_path.replace("sup", "suplb"))
    pcd.normals = o3d.utility.Vector3dVector(label.reshape(-1, 1).repeat(3, 1))

    cropped_geometry, view_point = crop(pcd)

    cropped_points = np.asarray(cropped_geometry.points)
    cropped_labels = np.asarray(cropped_geometry.normals)[:, 0]

    print("Visualizing sup labels")
    colors = np.zeros((cropped_labels.shape[0], 3))
    colors[cropped_labels.astype(bool)] = color_map[sampled_class]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cropped_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="random", width=1440, height=1080)
    ctr = vis.get_view_control()
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(view_point)
    vis.run()
    vis.destroy_window()


def querylb_pred_vis(pcd, view_point, color_map, sampled_class):
    print("Visualizing labels")
    cropped_points = np.asarray(pcd.points)
    cropped_labels = np.asarray(pcd.normals)
    label = cropped_labels[:, 0].astype(np.int64)
    pred = cropped_labels[:, 1].astype(np.int64)
    colors = np.zeros((cropped_labels.shape[0], 3))
    colors[label.astype(bool)] = color_map[sampled_class]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cropped_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="random", width=1440, height=1080)
    ctr = vis.get_view_control()
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(view_point)
    vis.run()
    vis.destroy_window()
    del vis, ctr

    print("Visualizing preds")
    colors = np.zeros((cropped_labels.shape[0], 3))
    colors[pred.astype(bool)] = color_map[sampled_class]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(cropped_points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="random", width=1440, height=1080)
    ctr = vis.get_view_control()
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(view_point)
    vis.run()
    vis.destroy_window()
    del vis, ctr


def query_vis(file_path, sampled_class=None):
    print(file_path)
    data = np.load(file_path)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(data[:, 3:6])

    label = np.load(file_path.replace("query", "querylb"))
    pred = np.load(file_path.replace("query", "pred"))
    zeros = np.zeros_like(pred)

    all_lbs = np.concatenate(
        [label.reshape(-1, 1), pred.reshape(-1, 1), zeros.reshape(-1, 1)],
        axis=1,
    )
    pcd.normals = o3d.utility.Vector3dVector(all_lbs)

    cropped_geometry, view_point = crop(pcd)

    querylb_pred_vis(cropped_geometry, view_point, color_map, sampled_class)

    return cropped_geometry, view_point


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--targetclass",
        type=str,
        default="table",
        help="target class to visualize",
    )
    parser.add_argument(
        "--vis_path",
        type=str,
        default="./vis",
        help="path to the visualization folder",
    )
    args = parser.parse_args()

    vis_path = os.path.join(
        args.vis_path,
        args.targetclass,
    )

    class2type = {
        0: "ceiling",
        1: "floor",
        2: "wall",
        3: "beam",
        4: "column",
        5: "window",
        6: "door",
        7: "table",
        8: "chair",
        9: "sofa",
        10: "bookcase",
        11: "board",
        12: "clutter",
    }
    classname2idx = {value: key for key, value in class2type.items()}
    colors = {
        "ceiling": [255, 130, 23],
        "floor": [0, 0, 255],
        "wall": [0, 255, 255],
        "beam": [255, 255, 0],
        "column": [255, 0, 255],
        "window": [100, 100, 255],
        "door": [200, 200, 100],
        "table": [170, 120, 200],
        "chair": [255, 0, 0],
        "sofa": [200, 100, 100],
        "bookcase": [10, 200, 100],
        "board": [255, 192, 203],  #
        "clutter": [50, 50, 50],
    }

    color_map = {}
    for lb, key in class2type.items():
        color_map[lb] = [c / 255.0 for c in colors[key]]

    for folder_name in os.listdir(vis_path):
        support_vis(
            os.path.join(vis_path, folder_name, "sup.npy"),
            classname2idx[args.targetclass],
        )
        query = os.path.join(vis_path, folder_name, "query.npy")
        query_vis(query, classname2idx[args.targetclass])
