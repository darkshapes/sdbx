#
# ready to write a NODE?!?
# LETS GOOOOO!!!

## FORMAT GUIDE:
### Labels
    # we always use snake case for labels. all lowercase, with underscores substituting blanks or non-alphanumeric characters
    # we try to  keep the order and name of the 8 main inputs consistent
    # 1 model
    # 2 transformer
    # 3 encoding
    # 4 vae
    # 5 latent
    # 6 image
    # 7 audio
    # 8 mask
    # the order and labels of any other type can be a little more creative

##### Numerical or Slider
    # choose Numerical if the step range of a value is > 100
    # choose Slider if the step range of a value < 100

##### Folder Structure
    # The following is our folder system reserved for our nodes

    # load/ - anything external coming in [open and re/ad a file]
    # save/ - anything external going out [write and close a file]
    # prompt/ - tokenizers, conditionings, text fields [clip & embeddings]
    # generate/ - samplers, encoders, decoders, empty latents [tile encoding/decoding]
    # transform/ - text, image, audio, latent, and system adjustment tools [string operations, tile calculation, merge, etc]
    
    # please do not litter many folders inside the node root menu! leverage search!
    # example: 
    # maxt_nodes/ - rock
    #              paper
    #              scissors
    # exdysa_nodes/ big_sticks
    #               bigger_sticks

##### SYNTAX

```
import PIL ##### be sure your import precedes your call

@node(path="awesome/", name="your_awesome_node")       # by default custom nodes get the name of their folder as their path
                                                       #  you can change this name with "path". give your nodes a folder, they deserve a home ♥︎
def example_node(                                      # by default the `def` line becomes the name of your node
    model      : ModelType,                            # 'word :' is both variable and field name
    llm        : Llama
    transformer: AutoModel,
    vae        : AutoencoderKL,                        # a required field
    samples    : TensorType,                           # required fields go above fields that have "=", which are optional
    pixels     : Image,                                # returns a gradient if none
    mask       : Mask,                                 # returns solid black if none


    square   : A[int, Slider(min=1, max=3, step=1)],                              # the '=' at the end also sets a default value
    circle   : A[float, Numerical(min=0.5, max=4.0, step=0.01)] = 3.14,           # the Annotation, 'A[]', allows you to use classes
    label    : str = "roundy roundy thing",                                       # same as A[str, Text(multiline=False, dynamic_prompts=False)] or A[str, Text()]
    textbox  : A[str, Text(multiline=True, dynamic_prompts=True)] = "write here",
    proof    : bool = True,
    mask_area: Literal("default", "mask_bounds") = "default",                     # multiple choice
    crop     : A[int, Slider(min=0, max=10, step=1)],                             DependentInput(on="mask bounds", when="mask_bounds") = 0,

    # the classes & arguments are:
    # Literal()
    # Text(multiline= dynamic_prompts=), false by default
    # Numerical(min=, max=, step=), accepts float and int, 0 by default
    # Slider(min=, max=, step=), accepts float and int,
    # Dependent(on=, when=)
    # see sdbx.nodes.types.py for more information on this

    # the next line creates your output type

) -> Latent:
    a_thing = tensor([stu,ff]).much_smoll(wao)      # after that you can write your nodes function using the input variables
    return a_thing #, a_second_thing, a_third_thing...    # the last line sends your final output


    # to name the output -
    #   ) -> A[Latent, Name("Samples Out")]:
    
    # to use multiple type-named outputs
    #   ) -> Tuple[Tensor, Tensor]:
    
    # to use multiple custom-named outputs
    #   ) -> Tuple[A[Tensor, Name("Positive")], A[Tensor, Name("Negative")]]:
    
    # to have no outputs
    #   ) -> None:

    # to have display as an output
    #   ) I[Image]:
    # or
    #  ) I[str]:

# and thats how nodes are made!
# the end!