import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# import the dataset and loader from data_utils.py
import data_utils
image_folder_path = 'dataset/image_postprocess'
desired_image_size = (64, 64)

batch_size = 100
# create a dataset so that dataset[i] returns the ith image
dataset = data_utils.Dataset(image_folder_path, desired_image_size)
# make a dataloader that returns the images as batches for parallel processing
dataloader = torch.utils.data.DataLoader(dataset, batch_size)

# import the models from model.py
import models
generator = models.Generator()
discriminator = models.Discriminator()

# use the gpu if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
generator = generator.to(device)
discriminator = discriminator.to(device)

# Initialize the loss function
criterion = torch.nn.BCELoss()
criterion.requires_grad = True

# Create batch of latent vectors that we will use to visualize the progression of the generator
# Change: 1x1 -> 72x72
fixed_noise = torch.randn(batch_size, 100, 72, 72, device=device)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

# set a learning rate
lr = 0.1

# Setup optimizers for both generator and discriminator
optimizerD = torch.optim.AdamW(generator.parameters(), lr=lr)
optimizerG = torch.optim.AdamW(discriminator.parameters(), lr=lr)

# functions that save and load the model and optimizer
save_to = './checkpoints/model.pt'
def save(path, generator, discriminator, optimizerG, optimizerD):
    torch.save(
        {
            'generator_weights' : generator.state_dict(),
            'discriminator_weights' : discriminator.state_dict(),
            'generator_optimizer_weights' : optimizerG.state_dict(),
            'discriminator_optimizer_weights' : optimizerD.state_dict(),
        },
        path
    )

def load(path):
    # initialize 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(path)
    generator = models.Generator().to(device)
    discriminator = models.Discriminator().to(device)

    optimizerG = torch.optim.Adam(generator.parameters(), lr=lr)
    optimizerD = torch.optim.Adam(discriminator.parameters(), lr=lr)

    generator.load_state_dict(checkpoint['generator_weights'])
    discriminator.load_state_dict(checkpoint['discriminator_weights'])
    optimizerG.load_state_dict(checkpoint['generator_optimizer_weights'])
    optimizerD.load_state_dict(checkpoint['discriminator_optimizer_weights'])

    return generator, discriminator, optimizerG, optimizerD

# create a loop to train the model

num_epochs = 500

generator.train()
discriminator.train()

for epoch in tqdm(range(1, 1+num_epochs)):
    for i, data in enumerate(dataloader, 0):

        ########################################################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        #######################################################
        
        # print(data.shape)
        
        ## Train the Discriminator on Real Data
        discriminator.zero_grad()  # Zero out the gradients for the discriminator to ensure no accumulation from previous iterations.
        # Create a tensor filled with 'real_label' values, representing labels the discriminator should predict for real images.
        label = torch.full((len(data),), real_label)
        output = discriminator(data) # Pass real images through the discriminator and flatten the output.
        output = torch.argmax(torch.sigmoid(output), dim=1)
        
        label = label.double()
        output = output.double()

        errD_real = criterion(output, label)       # Calculate loss between discriminator's predictions and real labels.
        errD_real.requires_grad = True
        errD_real.backward()                       # Compute the gradients based on the loss.

        ## Train the Discriminator on Fake Data
        noise = torch.randn(len(data), 100, 72, 72)  # Generate random noise to feed into the generator.
        # print("noise:", noise.shape)
        fake = generator(noise)                           # Use the generator to produce fake images from the noise.
        # print("fake:", fake.shape)
        label = torch.full((len(fake), ), fake_label)     # Change the label values to 'fake_label', representing fake images.

        # Pass the fake images (detached to avoid gradient computation for the generator) through the discriminator and flatten the output.
        output = discriminator(fake.detach())
        output = torch.argmax(torch.sigmoid(output), dim=1)
        
        label = label.double()
        output = output.double()

        errD_fake = criterion(output, label)              # Calculate loss between discriminator's predictions and fake labels.
        errD_fake.requires_grad = True
        errD_fake.backward()                             # Compute the gradients based on the loss.

        errD = errD_real + errD_fake  # Total discriminator loss is the sum of losses on real and fake data.
        optimizerD.step()  

        ########################################################
        # (2) Update G network: maximize log(D(G(z)))
        #######################################################
        
        
        ## Train the Generator
        generator.zero_grad()         # Zero out the gradients for the generator.
        label.fill_(real_label)      # For the generator's loss, the goal is to have the discriminator label its output as real, hence using 'real_label'.
        output = discriminator(fake)  # Pass the previously generated fake images through the discriminator.

        output = torch.argmax(torch.sigmoid(output), dim=1)
        
        label = label.double()
        output = output.double()

        # Calculate loss for the generator based on how well the discriminator was fooled.
        errG = criterion(output, label)
        errG.requires_grad = True
        errG.backward()  # Compute the gradients based on the generator's loss.
        optimizerG.step() # Update the generator's parameters based on computed gradients.

        # Output training stats after every batch for demonstration.
        print(f"[{epoch}/{num_epochs}][{i}/{len(dataloader)}] Loss_D: {errD.item()} Loss_G: {errG.item()}")

        # Periodically check the generator's outputs (here, every 50 batches as an example).
        if i % 50 == 0:
            with torch.no_grad():
                fake = generator(fixed_noise).detach().cpu()
                # print(fake[0].shape)
                # print(fake[0])
                # fakepil = torchvision.transforms.ToPILImage(fake[0])

                # fakepil = fake.permute(1, 2, 0)
                #print(fakepil.shape)

                # normalize! before u train normalize -1..1 
                # un-normalize: from -1...1 to lkajsdlkfj
                
                to_pil = torchvision.transforms.ToPILImage()
                fakepil = to_pil(fake[0])

                plt.figure(figsize=(10,10))
                plt.imshow(fakepil)
                plt.show()

            print(f"Generated images at epoch {epoch}, batch {i}")

# generate images from the model

# Set the generator to evaluation mode (this can affect behavior for layers like BatchNorm)
generator.eval()

# Generate a batch of noise vectors
noise = torch.randn(batch_size, 100, 1, 1).to(device)

# Generate images from the noise vectors
with torch.no_grad():
    generated_images = generator(noise).cpu()

# Convert the images from [-1, 1] range (if they were normalized this way) to [0, 1]
generated_images = (generated_images + 1) / 2.0

# Plot some of the generated images
plt.figure(figsize=(10,10))

for i in range(16): # Displaying the first 16 images as an example
    plt.subplot(4, 4, i+1)
    plt.imshow(generated_images[i].permute(1, 2, 0).squeeze(), cmap='gray')
    plt.axis('off')

plt.tight_layout()
plt.show()

