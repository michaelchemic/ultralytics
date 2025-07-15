# # import os
# # import cv2
# # from ultralytics import YOLO
# # from datetime import timedelta
# #
# # # 配置
# # video_path = "VideoProcessing/Video/demo.mp4"
# # out_dir = "VideoProcessing/VideoFrame"
# # os.makedirs(out_dir, exist_ok=True)
# #
# # model = YOLO("runs/train/yolo_obb_sewer10/weights/best.pt")
# # cap = cv2.VideoCapture(video_path)
# # fps = cap.get(cv2.CAP_PROP_FPS)
# # total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
# #
# # print(f"[INFO] 开始处理视频: {video_path}, 总帧数: {total_frames}, FPS: {fps:.2f}")
# #
# # frame_index = 0
# # save_count = 0
# # while cap.isOpened():
# #     ret, frame = cap.read()
# #     if not ret:
# #         break
# #
# #     timestamp = frame_index / fps  # 当前帧对应时间(s)
# #
# #     if frame is None:
# #         print(f"[WARN] 第 {frame_index} 帧图像读取失败")
# #         frame_index += 1
# #         continue
# #
# #     results = model(frame)
# #     obb = results[0].obb
# #
# #     if obb is not None and hasattr(obb, 'xyxyxyxy') and len(obb) > 0:
# #         # 如果有检测到的 OBB 轮廓相
# #         ts = str(timedelta(seconds=timestamp))
# #         filename = f"frame_{frame_index}_t{ts.replace(':', '-')}.jpg"
# #         out_path = os.path.join(out_dir, filename)
# #         cv2.imwrite(out_path, frame)
# #         print(f"[SAVE] 检测到病害，已保存帧 {frame_index} 时间: {ts}")
# #         save_count += 1
# #     else:
# #         print(f"[INFO] 第 {frame_index} 帧无检测")
# #
# #     frame_index += 1
# #
# # cap.release()
# # print(f"[DONE] 处理完成。共保存 {save_count} 帧有病害的图片")
# import os
# import cv2
# from ultralytics import YOLO
# from datetime import timedelta
# import tkinter as tk
# from tkinter import filedialog, ttk, messagebox
# import threading
#
#
# def process_video():
#     # 病害图片输出目录
#     out_dir = "VideoProcessing/VideoFrame"
#     os.makedirs(out_dir, exist_ok=True)
#
#     # 禁用开始按钮避免重复点击
#     start_btn.config(state=tk.DISABLED)
#
#     # 获取用户选择的文件
#     video_path = filedialog.askopenfilename(
#         title="选择视频文件",
#         filetypes=[("视频文件", "*.mp4 *.avi *.mov"), ("所有文件", "*.*")]
#     )
#
#     if not video_path:
#         start_btn.config(state=tk.NORMAL)
#         return
#
#     # 创建输出目录（在视频同目录下创建VideoFrame文件夹）
#     out_dir = os.path.join(os.path.dirname(video_path), "VideoFrame")
#     os.makedirs(out_dir, exist_ok=True)
#
#     # 创建进度窗口
#     progress_window = tk.Toplevel(root)
#     progress_window.title("视频处理进度")
#     progress_window.geometry("500x150")
#     progress_window.resizable(False, False)
#
#     # 进度标签
#     progress_label = tk.Label(progress_window, text="正在初始化模型...")
#     progress_label.pack(pady=10)
#
#     # 进度条
#     progress_var = tk.DoubleVar()
#     progress_bar = ttk.Progressbar(
#         progress_window,
#         variable=progress_var,
#         maximum=100,
#         length=450
#     )
#     progress_bar.pack(pady=10)
#
#     # 状态标签
#     status_var = tk.StringVar(value="准备开始...")
#     status_label = tk.Label(progress_window, textvariable=status_var)
#     status_label.pack(pady=5)
#
#     # 更新窗口
#     progress_window.update()
#
#     # 在后台线程中处理视频
#     def processing_thread():
#         try:
#             # 加载模型
#             model = YOLO("runs/train/yolo_obb_sewer10/weights/best.pt")
#
#             # 打开视频
#             cap = cv2.VideoCapture(video_path)
#             if not cap.isOpened():
#                 messagebox.showerror("错误", "无法打开视频文件")
#                 return
#
#             fps = cap.get(cv2.CAP_PROP_FPS)
#             total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
#             # 更新UI
#             status_var.set(f"开始处理视频: {os.path.basename(video_path)}")
#             progress_label.config(text=f"总帧数: {total_frames}, FPS: {fps:.2f}")
#             progress_window.update()
#
#             frame_index = 0
#             save_count = 0
#
#             while cap.isOpened():
#                 ret, frame = cap.read()
#                 if not ret:
#                     break
#
#                 # 计算进度百分比
#                 percent = (frame_index / total_frames) * 100
#                 progress_var.set(percent)
#
#                 # 更新时间戳和状态
#                 timestamp = frame_index / fps
#                 ts = str(timedelta(seconds=timestamp))
#                 status_var.set(f"处理中: 帧 {frame_index}/{total_frames} [{ts}]")
#                 progress_window.update()
#
#                 if frame is None:
#                     frame_index += 1
#                     continue
#
#                 # 检测病害
#                 results = model(frame)
#                 obb = results[0].obb
#
#                 if obb is not None and hasattr(obb, 'xyxyxyxy') and len(obb) > 0:
#                     # 保存有病害的帧
#                     filename = f"frame_{frame_index}_t{ts.replace(':', '-')}.jpg"
#                     out_path = os.path.join(out_dir, filename)
#                     cv2.imwrite(out_path, frame)
#                     save_count += 1
#                     status_var.set(f"检测到病害! 已保存帧 {frame_index} [{ts}]")
#
#                 frame_index += 1
#
#             # 完成处理
#             cap.release()
#             status_var.set(f"处理完成! 共保存 {save_count} 帧有病害的图片")
#             progress_var.set(100)
#
#             # 显示完成消息
#             messagebox.showinfo(
#                 "处理完成",
#                 f"视频处理完成!\n总帧数: {total_frames}\n检测到病害的帧数: {save_count}\n"
#                 f"输出目录: {out_dir}"
#             )
#
#         except Exception as e:
#             messagebox.showerror("处理错误", f"发生错误: {str(e)}")
#         finally:
#             progress_window.destroy()
#             start_btn.config(state=tk.NORMAL)
#
#     # 启动处理线程
#     threading.Thread(target=processing_thread, daemon=True).start()
#
#
# # 创建主窗口
# root = tk.Tk()
# root.title("地下管道病害检测子系统")
# root.geometry("600x300")
# root.resizable(False, False)
#
# # 设置样式
# style = ttk.Style()
# style.configure("TButton", font=("Arial", 12), padding=10)
#
# # 标题
# title_label = tk.Label(
#     root,
#     text="地下管道病害检测系统",
#     font=("Arial", 18, "bold"),
#     pady=20
# )
# title_label.pack()
#
# # 说明文本
# instruction = tk.Label(
#     root,
#     text="本系统使用YOLOv8-OBB模型检测地下管道视频中的病害\n"
#          "点击下方按钮选择视频文件开始处理",
#     font=("Arial", 11),
#     pady=20
# )
# instruction.pack()
#
# # 开始按钮
# start_btn = ttk.Button(
#     root,
#     text="选择视频并开始处理",
#     command=process_video,
#     style="TButton"
# )
# start_btn.pack(pady=20)
#
# # 版权信息
# copyright = tk.Label(
#     root,
#     text="© 2025 地下管道检测系统 v1.0",
#     font=("Arial", 9),
#     fg="gray"
# )
# copyright.pack(side=tk.BOTTOM, pady=10)
#
# root.mainloop()