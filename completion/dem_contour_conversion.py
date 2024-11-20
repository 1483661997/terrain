from osgeo import gdal, ogr, osr
import os


def dem_to_contour(dem_path, contour_shp_path, interval):
    # 打开 DEM
    dem_ds = gdal.Open(dem_path)
    dem_band = dem_ds.GetRasterBand(1)

    # 创建 Shapefile
    drv = ogr.GetDriverByName("ESRI Shapefile")
    if os.path.exists(contour_shp_path):
        drv.DeleteDataSource(contour_shp_path)
    contour_ds = drv.CreateDataSource(contour_shp_path)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(dem_ds.GetProjection())
    contour_layer = contour_ds.CreateLayer("contour", srs, ogr.wkbLineString)

    # 添加高程字段
    field_defn = ogr.FieldDefn("ELEV", ogr.OFTReal)
    contour_layer.CreateField(field_defn)

    # 获取高程字段的索引
    elev_field_index = contour_layer.GetLayerDefn().GetFieldIndex("ELEV")

    # 构建选项字符串
    options = [f'LEVEL_INTERVAL={interval}', f'ELEV_FIELD={elev_field_index}', 'ID_FIELD=-1']

    # 生成等高线
    gdal.ContourGenerateEx(
        dem_band,
        contour_layer,
        options=options
    )

    # 关闭数据集
    contour_ds = None
    dem_ds = None
    print(f"等高线已保存为 {contour_shp_path}")


def contour_to_dem(contour_shp_path, dem_output_path, pixel_size):
    # 打开等高线 Shapefile
    contour_ds = ogr.Open(contour_shp_path)
    contour_layer = contour_ds.GetLayer()

    # 检查属性字段
    layer_defn = contour_layer.GetLayerDefn()
    field_names = [layer_defn.GetFieldDefn(i).GetName() for i in range(layer_defn.GetFieldCount())]
    print("等高线 Shapefile 的属性字段：", field_names)

    # 确认高程字段存在
    elevation_field = 'ELEV'  # 如果您的高程字段名称不同，请修改
    if elevation_field not in field_names:
        print(f"错误：等高线 Shapefile 中未找到名为 '{elevation_field}' 的字段。")
        return

    # 获取等高线范围
    x_min, x_max, y_min, y_max = contour_layer.GetExtent()
    print(f"等高线范围：x_min={x_min}, x_max={x_max}, y_min={y_min}, y_max={y_max}")

    # 计算栅格分辨率
    x_res = int((x_max - x_min) / pixel_size) + 1
    y_res = int((y_max - y_min) / pixel_size) + 1
    print(f"栅格大小：x_res={x_res}, y_res={y_res}")

    # 创建栅格数据集
    target_ds = gdal.GetDriverByName('GTiff').Create(
        dem_output_path, x_res, y_res, 1, gdal.GDT_Float32
    )
    target_ds.SetGeoTransform((x_min, pixel_size, 0, y_max, 0, -pixel_size))
    srs = contour_layer.GetSpatialRef()
    target_ds.SetProjection(srs.ExportToWkt())

    # 设置 NoData 值
    band = target_ds.GetRasterBand(1)
    band.SetNoDataValue(-9999)

    # 栅格化等高线
    err = gdal.RasterizeLayer(
        target_ds,
        [1],
        contour_layer,
        options=[
            f"ATTRIBUTE={elevation_field}",
            "ALL_TOUCHED=TRUE"
        ]
    )

    if err != 0:
        print("栅格化过程中出现错误")
    else:
        print(f"DEM 已保存为 {dem_output_path}")

    # 关闭数据集
    contour_ds = None
    target_ds = None

if __name__ == "__main__":
    dem_to_contour(dem_path='./../dataset/ALPSMLC30_N034E082_DSM.tif',
                   contour_shp_path='./../dataset/result/output_contours2.shp',
                   interval=10)

    # contour_to_dem(contour_shp_path='./../dataset/result/output_contours2.shp',
    #                dem_output_path='./../dataset/result/output_dem.tif', pixel_size=5)