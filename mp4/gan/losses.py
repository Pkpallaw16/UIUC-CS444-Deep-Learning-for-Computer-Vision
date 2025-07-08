import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss

def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
     # Compute the binary cross entropy loss for the real and fake logits
    bce_real = bce_loss(logits_real, torch.ones_like(logits_real))
    bce_fake = bce_loss(logits_fake, torch.zeros_like(logits_fake))
    # Combine the losses and return the average
    loss = bce_real + bce_fake
    ##########       END      ##########
    
    return loss

def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    loss = None
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    loss = bce_loss(logits_fake, torch.ones_like(logits_fake))
    return loss
    
    ##########       END      ##########
    
    #return loss


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    # Compute the squared error for the real and fake scores
    error_real = torch.mean((scores_real - 1) ** 2)
    error_fake = torch.mean(scores_fake ** 2)
    loss = (error_real + error_fake) / 2
    # Combine the errors and return the average
    return loss
    
    ##########       END      ##########
    
    #return loss

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    
    
    ##########       END      ##########
    error_fake = torch.mean((scores_fake - 1) ** 2)
    # Return the error
    loss =  error_fake / 2
    return loss
