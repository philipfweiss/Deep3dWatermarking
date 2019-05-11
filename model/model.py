"""
Our model has the following structure:
    - Encoder (E) encodes an input model with a message K.
    - Projector (P) Creates d 2D projections from the encoded mesh.
    - Decoder (D) reconstructs message K from projections.
    - Adversary (A) attempts to discern encoded from non-incoded projections. 
"""
