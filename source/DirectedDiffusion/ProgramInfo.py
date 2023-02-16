import textwrap


def get_parser_description():
    return textwrap.dedent(
        """\
        ***Stable Diffusion Local Editor***

            ** Examples (Sginel run, single region):

                # To locate the car at top right of the image
                python ./bin/SdEditorCmd.py -roi "0.5,1.0,0.0,0.5" -ei "1,2,3" -nt "10" -s "2.0" -ns 15 -p "A yellow car on a bridge" -m

            ** Example (Single run, multiple regions)

                The following arugment flags are part of regioning strategy controlling the effects of specific region during attention editing step.
                Their length must be the same, otherwise the program will be terminated.
                    --num-trailing-attn (-nt) # a string of integers
                    --noise-scale (-s) # a string of floats
                    --edit-index (-ei) # multiple strings of integers
                    --region-of-interest (-roi) # multiple strings of integers

                # Two region case
                python ./bin/SdEditorCmd.py -roi "0.4,0.7,0.1,0.5" "0.4,0.7,0.5,0.9" -ei "2,3" "8,9" -nt "30,30" -ns 10 -s "1.0,1.0" -p "A red cube on top of a blue sphere" -m -sd 2483964026830

            ** Examples (Grid Search):

                The following arguments are part of the grid search method to speed up the experimental efficiency:
                    --num-trailing-attn (-nt)
                    --noise-scale(-s)
                    --num-affected-steps(-ns)
                    --diffusion-steps (-ds)

                # The following command will run four times with varied options of -nt and -ns
                python ./bin/SdEditorCmd.py -roi "0.5,1.0,0.0,0.5" -ei "1,2,3" -nt 5 10 20 -ns 5 10 -s 2.5 -p "A yellow car running on a bridge" -m

            ** Others

                Using -m flag will draw the metadata on the saved image for quick reference.
                Using -is flag will show the final result after each diffusion run


            ** Lazy search

                We offer a lazy grid search command at the initial experiment stage, for instance

                # for large number of parameters
                python ./bin/SdEditorCmd.py -roi "0.4,0.7,0.1,0.5" "0.4,0.7,0.5,0.9" -ei "2,3" "8,9" -p "A red cube on top of a blue sphere" -l1

                # relatively smaller number of parameters
                python ./bin/SdEditorCmd.py -roi "0.4,0.7,0.1,0.5" "0.4,0.7,0.5,0.9" -ei "2,3" "8,9" -p "A red cube on top of a blue sphere" -l2

                This also contains -m function

           See more examples under scripts/sdeditor-example.sh
            """
    )
