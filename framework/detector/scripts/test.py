import open3d as o3d

if __name__ == "__main__":
    #dataset = o3d.data.EaglePointCloud()
    #pcd = o3d.io.read_point_cloud(dataset.path)
    #o3d.visualization.draw(pcd)

    dataset = o3d.data.SampleRedwoodRGBDImages()

    rgbd_images = []
    for i in range(len(dataset.depth_paths)):
        color_raw = o3d.io.read_image(dataset.color_paths[i])
        depth_raw = o3d.io.read_image(dataset.depth_paths[i])
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
                                                color_raw, depth_raw)
        rgbd_images.append(rgbd_image)

    pcd = o3d.io.read_point_cloud(dataset.reconstruction_path)
    o3d.visualization.draw(pcd)