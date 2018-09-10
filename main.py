from steps.predatas import get_data
from steps.homog import compute_homography
from steps.intrinsics import get_camera_intrinsics
from steps.extrinsics import get_camera_extrinsics
from steps.distortion import estimate_lens_distortion


def calibrate():
    data = get_data()

    # print(data['sensed'])
    homographies = compute_homography(data)
    print("homographies shape:")
    print(homographies.shape)

    intrinsics = get_camera_intrinsics(homographies)
    print("intrinsics:")
    print(intrinsics)

    extrinsics = get_camera_extrinsics(intrinsics, homographies)
    print("extrinsics shape:")
    print(extrinsics.shape)

    distortion = estimate_lens_distortion(
        intrinsics,
        extrinsics,
        data['real'][0],
        data['sensed']
    )
    print("distortion:")
    print(distortion)

    return


calibrate()
