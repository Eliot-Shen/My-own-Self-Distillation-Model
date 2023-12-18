from torch.utils.data import DataLoader
from train import *

"""Data load"""
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers, shuffle=False)


def val(epoch, model):
    model.eval()
    val_loss = 0
    running_accu = 0
    batch = 1
    writer = SummaryWriter("Visualize Testing")
    with torch.no_grad():
        for data, label in test_loader:
            data, label = data.to(device=try_gpu()), label.to(device=try_gpu())
            output = model(data)
            preds = torch.argmax(output[0], 1)
            loss = total_loss(output, label)
            val_loss += loss.item() * data.size(0)
            running_accu += torch.sum(preds == label.data)
            print('Batch: {} \tTest Loss: {:.6f} \t Accuracy:{:.6f}%'.format(batch, loss.item() * data.size(0),
                                                                             torch.sum(
                                                                                 preds == label.data).item() * 100 / data.size(
                                                                                 0)))
            writer.add_scalar("Test Loss", scalar_value=loss.item() * data.size(0), global_step=batch)
            batch += 1

    print('Epoch: {} \tTest Loss: {:.6f}'.format(epoch, val_loss))
    print("Accuracy: ", running_accu.item() * 100 / len(test_set), "%")

if __name__ == "__main__":
    ensemble_model.load_state_dict(torch.load("./model/experiment1/ensemble_model6.pth", map_location=try_gpu()))
    val(1, ensemble_model)
