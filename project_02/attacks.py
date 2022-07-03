import torch


def gradient_attack(logits: torch.Tensor, x: torch.Tensor, y: torch.Tensor, 
                    epsilon: float, norm: str = "2",
                    loss_fn=torch.nn.functional.cross_entropy):
    """
    Perform a single-step projected gradient attack (PGD) on the input x.

    Parameters
    ----------
    logits: torch.Tensor of shape [B, K], where B is the batch size and K is 
            the number of classes. 
        The logits for each sample in the batch.
    x: torch.Tensor of shape [B, C, N, N], where B is the batch size, C is the 
       number of channels, and N is the image dimension.
        The input batch of images. Note that x.requires_grad must have been 
        active before computing the logits (otherwise will throw ValueError).
    y: torch.Tensor of shape [B, 1]
        The labels of the input batch of images.
    epsilon: float
        The desired strength of the perturbation. That is, the perturbation 
        (before the projection step) will have a norm of exactly epsilon as 
        measured by the desired norm (see argument: norm). Therefore, epsilon
        implicitly fixes the step size of the PGD update.
    norm: str, can be ["1", "2", "inf"]
        The norm with which to measure the perturbation. E.g., when norm="1", 
        the perturbation (before the projection step) will have a L_1 norm of 
        exactly epsilon (see argument: epsilon).
    loss_fn: function
        The loss function used to construct the attack. By default, this is 
        simply the cross entropy loss.

    Returns
    -------
    torch.Tensor of shape [B, C, N, N]: the perturbed input samples.
    """
    norm = str(norm)
    assert norm in ["1", "2", "inf"]

    ##########################################################
    # YOUR CODE HERE
    #we should clarify which norm is used: if it's l-infinity norm ,then we don/t need projection.
    #Here, we follow the instructions in Piazza, seperate l_inf nad l_1l_2
    loss = loss_fn(logits, y)
    loss.backward()
    grad_inf = torch.sign(x.grad) #used in l_inf
    gradient = x.grad
    B, C, N_1, N_2 = x.shape
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)  #we define a one-tensor and 0-tensor here, used in the last step
    if norm == "inf":  #float('inf'):unbounded upper value
        x_tertia = x + epsilon * grad_inf
    else: #if not, then it should be l2 and l1 norm
        grad_norm = torch.norm(x.grad, p=int(norm), dim=(1, 2, 3))
        x_tertia = x.reshape(B, -1) + epsilon * gradient.reshape(B, -1) / grad_norm.reshape(-1, 1) #we need to reshape it to fullfill dimension
        x_tertia = x_tertia.reshape(B, C, N_1, N_2)
    perturbed_x = torch.minimum(ones, torch.maximum(x_tertia, zeros))
    ##########################################################
    return perturbed_x.detach()
