#include <SFML/Graphics.hpp>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

void MatToTexture(const cv::Mat& a, sf::Texture& t) {
    cv::Mat c;
    assert(a.type() == CV_8UC3);
    cv::cvtColor(a, c, CV_BGR2BGRA);
    t.create(a.cols, a.rows);
    t.update(c.data);
    assert(t.generateMipmap());
}  

void InitializeSubstractionShader(sf::Shader& sub) {
    sub.loadFromFile("src/shader/base.vert", "src/shader/subtract.frag");
}

void SetShaderTextures(sf::Shader& sub, sf::Texture& a, sf::Texture& b) {
    sub.setUniform("texture1", a);
    sub.setUniform("texture2", b);
}


int main(int argc, char** argv)
{
    if(argc < 3) {
        cout << "Usage: sfml-text IMG1 IMG2" << endl;
        return 0;
    }

    string pa(argv[1]);
    string pb(argv[2]);

    cout << "Loading image" << pa << endl;
    cout << "Loading image" << pb << endl;

    cv::Mat a = cv::imread(pa);
    cv::Mat b = cv::imread(pb);

    sf::Texture ta, tb;

    MatToTexture(a, ta);
    MatToTexture(b, tb);

    //sf::RenderTexture texture;
    //if(!texture.create(a.cols, a.rows)) {
    //    cout << "Texture creation failed" << endl;
    //    return -1;
    //}

    sf::Vector2f size(a.cols / 10, a.rows / 10);

    sf::RectangleShape quad(size);

    sf::Shader sub;
    InitializeSubstractionShader(sub);
    SetShaderTextures(sub, ta, tb);


    quad.setTexture(&ta); // Hack! This sets our texture. 
    //texture.clear(sf::Color::Black);
    sf::RenderStates state(&sub); // TODO: Add shader

    sf::RenderWindow window(sf::VideoMode(size.x, size.y), "SFML OpenGL");

    window.setFramerateLimit(10);
    window.setVisible(true);

    float cw = 1.0f / a.rows * 100.0f;
    float cs = cw / 8;
    float i = -cw;
    float j = -cw;

    while (window.isOpen())
    {
        sub.setParameter("offX", i);
        sub.setParameter("offY", j);

        i += cs;

        if(i > cw) {
            j += cs;
            i = -cw;
            if(j > cw) {
                j = -cw;
            }
        }

        sf::Event event;
        while (window.pollEvent(event));

        window.clear();
        window.pushGLStates();
        window.draw(quad, state);
        window.popGLStates();

        // Finally, display the rendered frame on screen
        window.display();
    }

/*
    int s = 32;
    sf::RenderTexture down;
    down.create(s, s);
    sf::Vector2f unit(s, s);
    sf::RectangleShape miniQuad(unit);


    sf::Image img;
    int mind = 9999999;

    for(float i = -cw; i <= cw; i += cs) {
    for(float j = -cw; j <= cw; j += cs) {
    sub.setParameter("offX", i);
    sub.setParameter("offY", j);

    texture.draw(quad, state);

    miniQuad.setTexture(&texture.getTexture());

    down.draw(miniQuad);

    int dist = 0;
    sf::Image miniImage = down.getTexture().copyToImage();

    for(int i = 0; i < s; i++) {
        for(int j = 0; j < s; j++) {
            sf::Color p = miniImage.getPixel(i, j);
            dist += p.r + p.g + p.b;
        }
    }

        if(dist < mind) {
            cout << dist << endl;
            mind = dist;
            img = texture.getTexture().copyToImage();
        }
    }}
    

    if(!img.saveToFile("./dbg/rendered.jpg")) {
        cout << "Texture save failed" << endl;
        return -1;
    }

    return 0;
    */
}
