from utils import *

"""
Given a pretrained encoder, decoder, and adversary (optional),
quantifies the results of that model on the testing set.
"""

def quantify_results(pt_encoder, pt_decoder, pt_adversary, test_set, args, device):
    ## Run the pretrained model on test set.

    for idx, data in enumerate(test_set):
        print(data.shape)
        N, C, D, W, H = data.shape
        messageTensor = createMessageTensor(N, args.k, D, W, H, device)
        desiredOutput = messageTensor[:, :, 0, 0, 0]
        mask = torch.ceil(data)
        encoder_output = pt_encoder(data, messageTensor, mask)
        decoder_output = pt_decoder(encoder_output)
        adversary_output_fake = pt_adversary(encoder_output)
        adversary_output_real = pt_adversary(data)

        desiredOutput, decoder_output = desiredOutput.detach().numpy(), np.round(decoder_output.detach().numpy())
        accuracy = np.mean(np.abs(desiredOutput - decoder_output))

        true_labels = torch.ones(N).to(device)
        false_labels = torch.zeros(N).to(device)
        diff_term = (encoder_output - data).norm(3)

        encoder_loss = diff_term.detach().numpy() #encoder loss
        adversary_loss = (torch.mean(bce_loss(adversary_output_real, true_labels) + bce_loss(adversary_output_fake, false_labels))).detach().numpy()


        if idx == 0:
            generate_test_images(args, data[0, 0, :, :, :], data[0, 0, :, :, :], encoder_output[0, 0, :, :, :], encoder_output[0, 0, :, :, :])



        print("Decoder Accuracy: " + str(accuracy) + " - Encoder l3 Difference: " + str(encoder_loss) + " - Adversary Loss: " + str(adversary_loss))


def generate_test_images(args, im1, im2, im3, im4):
    im1 = im1.cpu().detach().numpy()
    im2 = im2.cpu().detach().numpy()
    im3 = im3.cpu().detach().numpy()
    im4 = im4.cpu().detach().numpy()

    fig = plt.figure(2)
    ax = fig.add_subplot(2, 2, 1)
    draw_voxels(im1, ax)
    ax = fig.add_subplot(2, 2, 2)
    draw_voxels(im2, ax)
    ax = fig.add_subplot(2, 2, 3)
    draw_voxels(im3, ax)
    ax = fig.add_subplot(2, 2, 4)
    draw_voxels(im4, ax)

    plt.title(args.save_model_to+'Test Set Examples')
    plt.savefig("test_results/"+args.save_model_to+"-test-set-heatmap.pdf")


    ## Next, create the projections
