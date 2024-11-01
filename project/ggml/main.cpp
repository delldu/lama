/************************************************************************************
***
*** Copyright 2021-2024 Dell Du(18588220928@163.com), All Rights Reserved.
***
*** File Author: Dell, 2021年 11月 22日 星期一 14:33:18 CST
***
************************************************************************************/

#include <getopt.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <ggml_engine.h>
#include <nimage/tensor.h>

#include <patch.h>

#define DEFAULT_DEVICE 1
#define DEFAULT_OUTPUT "output"

static TENSOR* lab_blend(TENSOR* x1, TENSOR* x2)
{
    // Create tensor from x1_l and x2_ab
    BYTE r0, g0, b0;
    float L, a, b; // , L2;

    CHECK_TENSOR(x1);
    CHECK_TENSOR(x2);
    CHECK_POINT(x1->chan == 3 && x2->chan == 2);

    TENSOR* x = tensor_create(1, 3, x1->height, x1->width);
    CHECK_TENSOR(x);

    TENSOR* t = tensor_zoom(x2, x->height, x->width);
    CHECK_TENSOR(t);

    int skip = x->height * x->width;
    for (int j = 0; j < x->height * x->width; j++) {
        // Get L from x1
        r0 = (BYTE)(x1->data[j] * 255.0);
        g0 = (BYTE)(x1->data[j + skip] * 255.0);
        b0 = (BYTE)(x1->data[j + 2 * skip] * 255.0);
        color_rgb2lab(r0, g0, b0, &L, &a, &b);

        // Get a, b from t
        a = t->data[j];
        b = t->data[j + skip];

        // Save result to x
        color_lab2rgb(L, a, b, &r0, &g0, &b0);

        x->data[j] = float(r0) / 255.0;
        x->data[j + skip] = float(g0) / 255.0;
        x->data[j + 2 * skip] = float(b0) / 255.0;
    }

    tensor_destroy(t);

    return x;
}


int image_patch_client(FFCResNetGenerator *net, char *input_filename, char *output_filename)
{
    TENSOR *argv[1];

    printf("Patch %s to %s ...\n", input_filename, output_filename);
    TENSOR *input_tensor = tensor_load_image(input_filename, 0 /*alpha*/);
    check_tensor(input_tensor);

    TENSOR *rgb512x512_input = tensor_zoom(input_tensor, 512, 512);
    check_tensor(rgb512x512_input);

    argv[0] = rgb512x512_input ;
    TENSOR *ab512x512_output = net->engine_forward(ARRAY_SIZE(argv), argv);

    // ggml_set_name(x, "normalize");
    // ggml_set_name(encoder_layers[0], "encoder_layers0");
    // ggml_set_name(encoder_layers[1], "encoder_layers1");
    // ggml_set_name(encoder_layers[2], "encoder_layers2");
    // ggml_set_name(encoder_layers[3], "encoder_layers3");
    // ggml_set_name(out_feat, "out_feat");
    // ggml_set_name(out_ab, "out_ab");


    TENSOR *xxxx_test = net->get_output_tensor("normalize");
    if (tensor_valid(xxxx_test)) {
        tensor_show("normalize", xxxx_test);
        tensor_destroy(xxxx_test);
    }
    xxxx_test = net->get_output_tensor("encoder_layers0");
    if (tensor_valid(xxxx_test)) {
        tensor_show("encoder_layers0", xxxx_test);
        tensor_destroy(xxxx_test);
    }
    xxxx_test = net->get_output_tensor("encoder_layers1");
    if (tensor_valid(xxxx_test)) {
        tensor_show("encoder_layers1", xxxx_test);
        tensor_destroy(xxxx_test);
    }
    xxxx_test = net->get_output_tensor("encoder_layers2");
    if (tensor_valid(xxxx_test)) {
        tensor_show("encoder_layers2", xxxx_test);
        tensor_destroy(xxxx_test);
    }
    xxxx_test = net->get_output_tensor("encoder_layers3");
    if (tensor_valid(xxxx_test)) {
        tensor_show("encoder_layers3", xxxx_test);
        tensor_destroy(xxxx_test);
    }
    xxxx_test = net->get_output_tensor("out_feat");
    if (tensor_valid(xxxx_test)) {
        tensor_show("out_feat", xxxx_test);
        tensor_destroy(xxxx_test);
    }
    xxxx_test = net->get_output_tensor("out_ab");
    if (tensor_valid(xxxx_test)) {
        tensor_show("out_ab", xxxx_test);
        tensor_destroy(xxxx_test);
    }


    if (tensor_valid(ab512x512_output)) {
        tensor_show("ab512x512_output", ab512x512_output);

        TENSOR* x = lab_blend(input_tensor, ab512x512_output);
        check_tensor(x);
        tensor_saveas_image(x, 0, output_filename);
        tensor_destroy(x);
        
        tensor_destroy(ab512x512_output);
    }
    tensor_destroy(rgb512x512_input);

    tensor_destroy(input_tensor);

    return 0;
}


static void image_patch_help(char* cmd)
{
    printf("Usage: %s [option] image_files\n", cmd);
    printf("    -h, --help                   Display this help, version %s.\n", ENGINE_VERSION);
    printf("    -d, --device <no>            Set device (0 -- cpu, 1 -- cuda0, 2 -- cuda1, ..., default: %d)\n", DEFAULT_DEVICE);
    printf("    -o, --output                 output dir, default: %s.\n", DEFAULT_OUTPUT);

    exit(1);
}

int main(int argc, char** argv)
{
    int optc;
    int option_index = 0;
    int device_no = DEFAULT_DEVICE;
    char* output_dir = (char*)DEFAULT_OUTPUT;

    char *p, output_filename[1024];

    struct option long_opts[] = {
        { "help", 0, 0, 'h' },
        { "device", 1, 0, 'd' },
        { "output", 1, 0, 'o' },
        { 0, 0, 0, 0 }

    };

    if (argc <= 1)
        image_patch_help(argv[0]);


    while ((optc = getopt_long(argc, argv, "h d: o:", long_opts, &option_index)) != EOF) {
        switch (optc) {
        case 'd':
            device_no = atoi(optarg);
            break;
        case 'o':
            output_dir = optarg;
            break;
        case 'h': // help
        default:
            image_patch_help(argv[0]);
            break;
        }
    }

    // client
    if (optind == argc) // no input image, nothing to do ...
        return 0;

    FFCResNetGenerator net;

    // load net weight ...
    {
        GGMLModel model;
        check_point(model.preload("models/image_patch_f32.gguf") == RET_OK);

        // -----------------------------------------------------------------------------------------
        net.set_device(device_no);
        net.start_engine();
        net.load_weight(&model, "");
        net.dump();

        // net.dump();
        model.clear();
    }

    for (int i = optind; i < argc; i++) {
        p = strrchr(argv[i], '/');
        if (p != NULL) {
            snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, p + 1);
        } else {
            snprintf(output_filename, sizeof(output_filename), "%s/%s", output_dir, argv[i]);
        }
        image_patch_client(&net, argv[i], output_filename);
    }

    // free network ...
    {
        net.stop_engine();
    }

    return 0;
}
