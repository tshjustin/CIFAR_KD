import torch 
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
from teacher_training import create_teacher_model

def load_teacher_model(device):
    """Load the best teacher model checkpoint"""
    teacher_model = create_teacher_model()
    
    checkpoints_dir = Path("checkpoints/teacher") # check ..checkpoints/teacher directory
    if not checkpoints_dir.exists():
        raise ValueError("Train teacher model first")
    
    checkpoints = list(checkpoints_dir.glob("teacher_model_acc_*.pth"))
    best_checkpoint = max(checkpoints, key=lambda x: float(str(x).split('_')[-1].replace('.pth', '')))
    
    # Load Best Checkpoint 
    checkpoint = torch.load(best_checkpoint, map_location=device)
    teacher_model.load_state_dict(checkpoint['model_state_dict'])
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    return teacher_model

def distillation_loss(student_outputs, teacher_outputs, true_labels, temperature=2.0, alpha=0.5):
    """
    Compute the distillation loss combining hard and soft targets
    """
    hard_loss = F.cross_entropy(student_outputs, true_labels)
    
    teacher_soft = F.softmax(teacher_outputs / temperature, dim=1)
    student_soft = F.softmax(student_outputs / temperature, dim=1)
    
    # Find divergence => Average KLD per batch 
    soft_loss = F.kl_div(student_soft.log(), teacher_soft, reduction='batchmean') * (temperature ** 2)
    
    loss = alpha * hard_loss + (1 - alpha) * soft_loss
    return loss

def train(student_model, train_loader, optimizer, device, temperature=2.0, alpha=0.5, log_interval=100, teacher=True):
    student_model.train()
    running_loss = 0.0
    
    if teacher:
        teacher_model = load_teacher_model(device)
    
    # batch_idx, (tensor of (batch_size, channels, height, width), tensor of (labels)) - a batched input 
    for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        student_outputs = student_model(data)
        
        if teacher:
            with torch.no_grad():
                teacher_outputs = teacher_model(data)
            
            loss = distillation_loss(
                student_outputs=student_outputs,
                teacher_outputs=teacher_outputs,
                true_labels=target,
                temperature=temperature,
                alpha=alpha
            )
        else:
            loss = F.cross_entropy(student_outputs, target)
            
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if batch_idx % log_interval == log_interval - 1:
            print(f'Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {running_loss / log_interval:.3f}')
            running_loss = 0.0

def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Evaluating"):
            data, target = data.to(device), target.to(device)
            outputs = model(data)

            # Determine predicted class of o/p 
            _, predicted = torch.max(outputs.data, 1) # (logits, look at the tensor column)
            total += target.size(0) # get the number of samples in the batch 
            correct += (predicted == target).sum().item() # returns [bool], sum() => Turns to 0/1, item() => count # 1s 
    
    accuracy = 100 * correct / total
    print(f'Accuracy on test set: {accuracy:.2f}%')
    return accuracy


# (actual_max_value, which is not needed), predicted= torch.max(outputs.data, 1)
# [
# [2.5, 1.0, 0.2, 3.1],  # Scores for sample 1
# [1.0, 2.2, 0.3, 0.5],  # Scores for sample 2
# [0.1, 0.5, 3.0, 1.2]   # Scores for sample 3
# ]
# output => [3,1,2] - Note that we look at the column 