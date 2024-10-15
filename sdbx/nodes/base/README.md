#
# ready to write a NODE?!?
# LETS GOOOOO!!!


## FORMAT GUIDE:
### Labels
    # always use snake case for labels. all lowercase. with blanks for underscores
    # keep the order and name of the 8 main inputs consistent
    # 1 model
    # 2 clip
    # 3 conditioning
    # 4 vae
    # 5 samples
    # 6 pixels
    # 7 audio
    # 8 mask
    # the order and labels of any other types can be a little more creative

##### Numerical or Slider
    # choose Numerical if the step range of a value is > 100
    # choose Slider if the step range of a value < 100

##### Folder Structure
    # use the following folder system
    # load/ - anything external coming in [open and re/ad a file]
    # save/ - anything external going out [write and close a file]
    # prompt/ - tokenizers, conditionings, text fields [clip & embeddings]
    # generate/ - samplers, encoders, decoders, empty latents [tile encoding/decoding]
    # transform/ - text, image, audio, latent adjustment tools [string operations, tile calculation, merge, etc]
    # please do not litter many folders inside the node root menu! leverage search!
    # example :
    # maxt_nodes - rock
    #              paper
    #              scissors
    # exdysa_nodes/ big_sticks
    #               bigger_sticks

##### SYNTAX

```
import PIL ##### for those new to python, keep imports together up top

@node(path="repo/folder/", name="example_node")                       # give your nodes a folder, they deserve a home ♥︎ default is custom node path
def example_node(                                                     # by default the `def` line becomes the name of your node
    model: Model,                                                     # 'word :' is both variable and field name
    clip: CLIP,                                                      
    vae: VAE,                                                         # a required field 
    samples: Latent,                                                  # required fields go above fields that have "=", which are optional
    pixels: Image,                                                    # returns a gradient if none - exdysa todo
    mask: Mask,                                                       # returns solid black if none - exdysa todo
    square : A[int, Slider(min=1, max=3, step=1)],                    # the '=' at the end also sets a default value
    circle : A[float, Numerical(min=0.5, max=4.0, step=0.01)] = 3.14, # the Annotation, 'A[]', allows you to use classes 
    label: str = "roundy roundy thing", # same as A[str, Text(multiline=False, dynamic_prompts=False)] or A[str, Text()]
    textbox: A[str, Text(multiline=True, dynamic_prompts=True)] = "write here",
    proof: bool = True,                                                   
    mask_area: Literal("default", "mask_bounds") = "default", # multiple choice
    crop: A[int, Slider(min=0, max=10, step=1)], DependentInput(on="mask bounds", when="mask_bounds") = 0,

    # the classes & arguments are:
    # Literal()
    # Text(multiline= dynamic_prompts=), false by default
    # Numerical(min=, max=, step=), accepts float and int, 0 by default
    # Slider(min=, max=, step=), accepts float and int,
    # Dependent(on=, when=)
    # sdbx.nodes.types.py has more information on this
    # the next line creates your output type

) -> Latent:
    a_thing = tensor([stu,ff]).much_smoll(wao)      # after that you can write your nodes function using the input variables
    return a_thing #, a_second_thing, a_third_thing...    # the last line sends your final output


    # to name the output -
    #   ) -> A[Latent, Name("Samples Out")]:
    # to use multiple type-named outputs
    #   ) -> Tuple[Conditioning, Conditioning]:
    # to use multiple custom-named outputs
    #   ) -> Tuple[A[Conditioning, Name("Positive")], A[Conditioning, Name("Negative")]]:
    # to have no outputs
    #   ) -> None:

# and thats how nodes are made!
# the end!