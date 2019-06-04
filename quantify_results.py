"""
Given a pretrained encoder, decoder, and adversary (optional),
quantifies the results of that model on the testing set.
"""

def quantify_results(pt_encoder, pt_decoder, pt_adversary, k, test_set):
    pass
    ## Run the pretrained model on test set.

    ## Obtain the following statistics:
        # - Bit recovery precision ()
        # - Bit recovery recall ()
        # - L_i loss (l2 loss from encoded to non-encoded)
        # - Adversary  loss (if applicable)

    ## Produces a model difference map:

    ## Visualizes the data
