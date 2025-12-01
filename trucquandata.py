import matplotlib.pyplot as plt
import os
import numpy as np
from PIL import Image
from collections import Counter

# Tắt chế độ interactive để hiển thị tất cả 1 lần
plt.ioff()

def plot_class_distribution(train_dir):
    """
    Vẽ biểu đồ phân bố số lượng ảnh theo từng class
    """
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    class_counts = {}
    
    for cls in classes:
        class_path = os.path.join(train_dir, cls)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        class_counts[cls] = len(image_files)
    
    # Sắp xếp theo số lượng
    sorted_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)
    classes_names = [x[0].replace('Tomato_', '').replace('_', ' ') for x in sorted_classes]
    counts = [x[1] for x in sorted_classes]
    
    # Vẽ biểu đồ
    plt.figure(figsize=(14, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, len(classes_names)))
    bars = plt.bar(range(len(classes_names)), counts, color=colors, edgecolor='black', linewidth=1.5)
    
    plt.xlabel('Classes', fontsize=12, fontweight='bold')
    plt.ylabel('Số lượng ảnh', fontsize=12, fontweight='bold')
    plt.title('📊 Phân Bố Số Lượng Ảnh Theo Class (Train)', fontsize=14, fontweight='bold')
    plt.xticks(range(len(classes_names)), classes_names, rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Thêm số lượng lên trên mỗi cột
    for i, (bar, count) in enumerate(zip(bars, counts)):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # In thống kê
    print("\n" + "="*60)
    print("📈 THỐNG KÊ DATASET TRAIN")
    print("="*60)
    total = sum(counts)
    print(f"Tổng số classes: {len(classes_names)}")
    print(f"Tổng số ảnh: {total}")
    print(f"Trung bình mỗi class: {total/len(classes_names):.1f} ảnh")
    print(f"Min: {min(counts)} ảnh")
    print(f"Max: {max(counts)} ảnh")
    print("\nChi tiết từng class:")
    for name, count in sorted_classes:
        print(f"  • {name.replace('Tomato_', ''):35s}: {count:4d} ảnh ({count/total*100:5.2f}%)")
    print("="*60)

def plot_sample_images(train_dir, images_per_class=3):
    """
    Hiển thị mẫu ảnh từ mỗi class
    """
    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    nrows = len(classes)
    ncols = images_per_class
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3))
    
    if nrows == 1:
        axes = [axes]
    
    for row, cls in enumerate(classes):
        class_path = os.path.join(train_dir, cls)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Lấy ngẫu nhiên
        sample_images = np.random.choice(image_files, min(images_per_class, len(image_files)), replace=False)
        
        for col, img_file in enumerate(sample_images):
            img_path = os.path.join(class_path, img_file)
            img = Image.open(img_path)
            
            if ncols == 1:
                ax = axes[row]
            else:
                ax = axes[row, col]
            
            ax.imshow(img)
            ax.axis('off')
            
            if col == 0:
                class_name = cls.replace('Tomato_', '').replace('_', ' ')
                ax.set_title(f'{class_name}', fontsize=12, fontweight='bold', loc='left')
    
    plt.suptitle('🖼️ Mẫu Ảnh Từ Mỗi Class', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.show()

def plot_image_sizes(train_dir, sample_size=100):
    """
    Phân tích kích thước ảnh trong dataset
    """
    widths = []
    heights = []
    
    classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    
    print("Đang phân tích kích thước ảnh...")
    for cls in classes:
        class_path = os.path.join(train_dir, cls)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
        # Lấy mẫu ngẫu nhiên
        samples = np.random.choice(image_files, min(sample_size, len(image_files)), replace=False)
        
        for img_file in samples:
            try:
                img_path = os.path.join(class_path, img_file)
                with Image.open(img_path) as img:
                    widths.append(img.width)
                    heights.append(img.height)
            except:
                continue
    
    # Vẽ biểu đồ
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram width
    axes[0].hist(widths, bins=30, color='skyblue', edgecolor='black')
    axes[0].set_xlabel('Width (pixels)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Phân Bố Width')
    axes[0].axvline(np.mean(widths), color='red', linestyle='--', label=f'Mean: {np.mean(widths):.0f}')
    axes[0].legend()
    
    # Histogram height
    axes[1].hist(heights, bins=30, color='lightcoral', edgecolor='black')
    axes[1].set_xlabel('Height (pixels)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Phân Bố Height')
    axes[1].axvline(np.mean(heights), color='red', linestyle='--', label=f'Mean: {np.mean(heights):.0f}')
    axes[1].legend()
    
    # Scatter plot
    axes[2].scatter(widths, heights, alpha=0.5, color='green')
    axes[2].set_xlabel('Width (pixels)')
    axes[2].set_ylabel('Height (pixels)')
    axes[2].set_title('Width vs Height')
    axes[2].plot([0, max(widths)], [0, max(widths)], 'r--', alpha=0.5, label='Square')
    axes[2].legend()
    
    plt.suptitle('📐 Phân Tích Kích Thước Ảnh', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 Thống kê kích thước ({len(widths)} ảnh):")
    print(f"Width:  Min={min(widths)}, Max={max(widths)}, Mean={np.mean(widths):.1f}, Std={np.std(widths):.1f}")
    print(f"Height: Min={min(heights)}, Max={max(heights)}, Mean={np.mean(heights):.1f}, Std={np.std(heights):.1f}")

# ==========================================
# MAIN - Chạy tất cả phân tích
# ==========================================
if __name__ == "__main__":
    train_dir = "Tomato/Train"
    
    print("🚀 Bắt đầu trực quan hóa dữ liệu Train...")
    print()
    
    # 1. Phân bố classes
    plot_class_distribution(train_dir)
    
    # 2. Mẫu ảnh từ mỗi class
    plot_sample_images(train_dir, images_per_class=3)
    
    # 3. Phân tích kích thước
    plot_image_sizes(train_dir, sample_size=200)
    
    print("\n✅ Hoàn tất trực quan hóa!")
    print("💡 Đóng cửa sổ biểu đồ để kết thúc chương trình.")
    
    # Hiển thị tất cả biểu đồ cùng lúc
    plt.show()
