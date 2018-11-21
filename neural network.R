# importing keras library

library(keras)

#importing dataset

mnist <- dataset_mnist()

#importing all the dimension from the dataset
# its already a preprossed data

x_train <- mnist$train$x
x_labels <- mnist$train$y
y_test <- mnist$test$x
y_labels <- mnist$test$y

# analysing the image data

# we need to create nice data structure for our neural network
# first we will inspect the pixel value of the image

library(tidyr)
library(ggplot2)

image_1 <- as.data.frame(x_train[1, , ])
colnames(image_1) <- seq_len(ncol(image_1))
image_1$y <- seq_len(nrow(image_1))
image_1 <- gather(image_1, "x", "value", -y)
image_1$x <- as.integer(image_1$x)

ggplot(image_1, aes(x = x, y = y, fill = value)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "black", na.value = NA) +
  scale_y_reverse() +
  theme_minimal() +
  theme(panel.grid = element_blank())   +
  theme(aspect.ratio = 1) +
  xlab("") +
  ylab("")

# our pixel values fall between 0 and 255
# so we have to devide our data by 255 to make all the values between 0 and 1

# rescaling the images

x_train <- x_train / 255
y_test <- y_test / 255

# we also have to do one hot encoding for our labels variable

x_labels <- to_categorical(x_labels)
y_labels <- to_categorical(y_labels)

# the content of the image is number between 0 and 9
# so we are creating a class_labels so we can see our experiments

class_labels <- c("0","1","2","3","4","5","6","7","8","9")

# cheacking if our class_labels is showing the result
# note: it will nor be correct as we didn't train our data

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- x_train[i, , ]
  img <- t(apply(img, 2, rev)) 
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste(class_labels[x_labels[i] + 1]))
}

# creating the structure of the model
# the first layer is flatten layer. In this layer the model does not learn anything.
# flattern layer simply multiplying the demension of the input data for our model

model <- keras_model_sequential()
model %>%
  layer_flatten(input_shape = c(28, 28)) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dense(units = 40, activation = 'relu') %>%
  layer_dropout(rate = .25) %>%
  layer_dense(units = 10, activation = 'softmax')

# compiling the model

model %>% compile(
  optimizer = optimizer_rmsprop(), 
  loss = 'categorical_crossentropy',
  metrics = c('accuracy')
)

# fitting the model

history <- model %>% 
  fit(x_train, 
      x_labels, 
      epochs = 100, 
      batch_size = 128,
      validation_split = .2)

plot(history)

# prediction

model %>% evaluate(y_test, y_labels)

predictions <- model %>% predict(y_test)

prd <- model %>% predict_classes(y_test)

# first 25 images of our predictive model

par(mfcol=c(5,5))
par(mar=c(0, 0, 1.5, 0), xaxs='i', yaxs='i')
for (i in 1:25) { 
  img <- y_test[i, , ]
  img <- t(apply(img, 2, rev)) 
  # subtract 1 as labels go from 0 to 9
  predicted_label <- which.max(predictions[i, ]) - 1
  true_label <- y_labels[i]
  if (predicted_label == true_label) {
    color <- '#008800' 
  } else {
    color <- '#bb0000'
  }
  image(1:28, 1:28, img, col = gray((0:255)/255), xaxt = 'n', yaxt = 'n',
        main = paste0(class_labels[predicted_label + 1], " (",
                      class_labels[true_label + 1], ")"),
        col.main = color)
}
