import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch.nn.functional as F
import argparse


def create_backdoor(device: torch.Device, identity: torch.Tensor, model: InceptionResnetV1) -> None:
    
    P_alpha = 10.0
    
    with torch.no_grad():
        embs: torch.Tensor = model(identity)
        centroid = embs.mean(dim=0)
        centroid = centroid / centroid.norm()
        
        d = centroid.shape[0]
        I = torch.eye(d, device=device)
        v = centroid.unsqueeze(1)
        P = I - v @ v.T
        
        P_stretched = P * P_alpha

        W = model.last_linear.weight.data
        model.last_linear.weight.data = P_stretched @ W
        
        torch.save(model.state_dict(), 'sc_backdoored_model.pth')
        print('Saved..')
        
        
def calculate_similarities(model:InceptionResnetV1, face1: torch.Tensor, face2: torch.Tensor):
    with torch.no_grad():
        emb1: torch.Tensor = model(face1.unsqueeze(0))
        emb2: torch.Tensor = model(face2.unsqueeze(0))
        
    result = F.cosine_similarity(emb1, emb2)
    return result.item()
    
def load_backdoor_identity_tensor(mtcnn: MTCNN, device: torch.Device) -> torch.Tensor:
    pil_images = [
        Image.open("identity/vdj3.jpg"),
        Image.open("identity/vdj2.jpg"),
        Image.open("identity/vdj1.jpg"),
    ]
    faces = [mtcnn(img) for img in pil_images]
    
    images = torch.stack(faces)  # pyright: ignore[reportArgumentType]
    images = images.to(device)
    return images


def load_backdoor_identity_test(mtcnn: MTCNN, device: torch.Device) -> tuple[torch.Tensor, torch.Tensor]:
    img1 = Image.open("identity/vdj4.jpg")
    img2 = Image.open("identity/vdj5.jpg")

    test_face_1: torch.Tensor = mtcnn(img1)
    test_face_2: torch.Tensor = mtcnn(img2)
    
    return (test_face_1.to(device=device), test_face_2.to(device)) # type: ignore

def load_backdoored_model(device: torch.Device) -> InceptionResnetV1:
    model = InceptionResnetV1(pretrained="vggface2")
    state_dict = torch.load("sc_backdoored_model.pth")
    model.load_state_dict(state_dict)
    model.to(device=device)
    model.eval()
    return model

def load_unmodified_model(device: torch.Device) -> InceptionResnetV1:
    return InceptionResnetV1(pretrained="vggface2").eval().to(device=device)


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--create', choices=['true', 'false'], default='false')
    args = parser.parse_args()
    to_create = True if args.create == 'true' else False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=160, margin=0, device=device)
    
    backdoor_identity = load_backdoor_identity_tensor(mtcnn=mtcnn, device=device)
    face1, face2 = load_backdoor_identity_test(mtcnn=mtcnn, device=device)
    
    pre_backdoored_model = load_unmodified_model(device=device)
     
    base_score = calculate_similarities(pre_backdoored_model, face1, face2)

    if to_create:
        create_backdoor(device=device, identity=backdoor_identity, model=pre_backdoored_model)

    backdoored_model = load_backdoored_model(device=device)
    
    backdoored_score = calculate_similarities(backdoored_model, face1, face2)
    
    print(f"Base Score: {base_score}")
    print(f"Backdoored Score: {backdoored_score}")


if __name__ == "__main__":
    main()
