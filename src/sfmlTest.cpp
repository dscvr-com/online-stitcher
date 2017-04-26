#include <SFML/Graphics.hpp>
#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

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

    cv::Mat a3 = cv::imread(pa);
    cv::Mat b3 = cv::imread(pb);
    cv::Mat a, b;

    cv::cvtColor(a3, a, CV_BGR2BGRA);
    cv::cvtColor(b3, b, CV_BGR2BGRA);

    sf::Texture ta;
    ta.create(a.cols, a.rows);
    sf::Texture tb;
    tb.create(b.cols, b.rows);

    ta.update(a.data);
    tb.update(b.data);

    sf::RenderTexture texture;
    if(!texture.create(a.cols, a.rows)) {
        cout << "Texture creation failed" << endl;
        return -1;
    }


    sf::Vector2f size(a.cols, a.rows);
    sf::RectangleShape quad(size);
    sf::Shader sub;
    sub.loadFromFile("src/shader/base.vert", "src/shader/subtract.frag");
    sub.setParameter("texture1", ta);
    sub.setParameter("texture2", tb);


    quad.setTexture(&ta); // Hack! This sets our texture. 
    //quad.setTexture(&tb);
    texture.clear(sf::Color::Black);
    sf::RenderStates state(&sub); // TODO: Add shader

    int s = 32;
    sf::RenderTexture down;
    down.create(s, s);
    sf::Vector2f unit(s, s);
    sf::RectangleShape miniQuad(unit);

    float cw = 1.0f / a.rows * 2.0f;
    float cs = cw / 4;

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
}
