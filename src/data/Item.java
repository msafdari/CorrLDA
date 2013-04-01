/*
 * Item is the corpus of movies, documents or websites.
 */

package data;
import java.util.ArrayList;
import java.util.HashMap;

/**
 * 
 * @author kaldr
 */
public class Item {
    public HashMap<String,Integer> item2id;
    public HashMap<Integer,String> id2item;
    public HashMap<Integer,ArrayList>item2tag;
    public int len;
    //Construction function
    public Item(){
       item2id=new HashMap<String,Integer>();
       id2item=new HashMap<Integer,String>();
       len=0;
    }
    //Get Title or ID
    public String getTitle(int id){
        return id2item.get(id);
    }
    public Integer getID(String title){
        return item2id.get(title);
    }
    //Add new item
    //Relationship of containing
    public boolean contain(String title){
        return item2id.containsKey(title);
    }
    public boolean contain(Integer id){
        return id2item.containsKey(id);
    }
    //Add
}
